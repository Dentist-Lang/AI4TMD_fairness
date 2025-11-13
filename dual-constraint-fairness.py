import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader_fairness import CustomDataset
import numpy as np
import torchvision.models as models
import fairness_function_new
from statsmodels.stats.proportion import proportion_confint
from sklearn.utils import resample
from fairness_function_adv import adversarial_loss, Adversary

torch.set_num_threads(8)

fairness_method = 'spatial_alignment'
writer = open("dual-constraint-fairness.txt", "w+")  # 改为 task2



def setup_seed(seed):
    import numpy as np
    import random
    from torch.backends import cudnn
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

setup_seed(0)

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor()
])

# 文件路径配置
root_folder = ""
data_folder = ""
label_folder = ""
train_file = os.path.join(label_folder, "train.txt")
test_file = os.path.join(label_folder, "test.txt")


train_dataset = CustomDataset_mm_Group_3(label_folder=label_folder, data_folder=data_folder,
                              file=train_file, transform=transform, mode="train")
test_dataset = CustomDataset_mm_Group_3(label_folder=label_folder, data_folder=data_folder,
                             file=test_file, transform=transform, mode="test")

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SingleTaskModel(nn.Module):
    def __init__(self):
        super(SingleTaskModel, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        self.classifier = nn.Linear(2048 * 3, 3)

    def forward(self, pd_images, t2wi_images, zk_images):
        pd_features = [self.features(img.to(device)) for img in pd_images]
        t2wi_features = [self.features(img.to(device)) for img in t2wi_images]
        zk_features = [self.features(img.to(device)) for img in zk_images]

        pd_features = torch.mean(torch.stack(pd_features), dim=0)
        t2wi_features = torch.mean(torch.stack(t2wi_features), dim=0)
        zk_features = torch.mean(torch.stack(zk_features), dim=0)

        final_feature = torch.cat([pd_features, t2wi_features, zk_features], dim=1)
        final_feature = final_feature.view(final_feature.size(0), -1)

        task2_output = self.classifier(final_feature)
        return task2_output, final_feature

criterion_task2 = nn.CrossEntropyLoss()
criterion_adv = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SingleTaskModel()

adversary_model = Adversary(num_sensitive_classes=2).to(device)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)

optimizer = optim.Adam(
    list(model.parameters()) + list(adversary_model.parameters()),
    lr=0.0001
)

best_auc_task2 = 0
best_accuracy_task2 = 0

checkpoint_folder = "./checkpoint"
if not os.path.exists(checkpoint_folder):
    os.mkdir(checkpoint_folder)

num_epochs = 80

def plot_roc_curve(test_labels_binarized, preds_probs, num_classes, epoch):
    """
    test_labels_binarized: shape (N, num_classes), 例如 label_binarize(y, classes=[0,1,2])
    preds_probs: shape (N, num_classes), softmax 概率
    num_classes: 3
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, roc_auc_dict = {}, {}, {}
    plt.figure()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels_binarized[:, i], preds_probs[:, i])
        roc_auc_dict[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc_dict[i]:.2f})")

train_losses = []
test_losses = []

for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0
    train_task2_labels, train_task2_preds = [], []
    train_task2_outputs = []
    total_batches = len(train_loader)


    group_metrics = {
        'Gender': {
            0: {'pred_logits': [], 'label': []},  # Gender 0
            1: {'pred_logits': [], 'label': []}  # Gender 1
        },
    }
    avg_loss_adv = 0

    for index, (closed_pd_images, closed_t2w_images, open_pd_images, label2, gender) in enumerate(train_loader):

        label2 = label2.to(device)

        optimizer.zero_grad()
        task2_output, final_feature = model(closed_pd_images, closed_t2w_images, open_pd_images)
        loss_task2 = criterion_task2(task2_output, label2)
        loss_fairness = 0

        if fairness_method == 'spatial_alignment':
            loss_fairness = fairness_function_new.spatial_alignment_loss(task2_output, label2, gender.unsqueeze(0))
        else:
            raise TypeError

        loss_adv = adversarial_loss(final_feature, gender, adversary_model, criterion_adv)
        avg_loss_adv = avg_loss_adv + loss_adv

        loss = loss_task2 + loss_fairness + loss_adv

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, task2_preds_batch = torch.max(task2_output, 1)
        train_task2_labels.extend(label2.cpu().numpy())
        train_task2_preds.extend(task2_preds_batch.cpu().numpy())

        train_task2_outputs.append(task2_output.detach().cpu().numpy())

    avg_train_loss = train_loss / total_batches
    train_losses.append(avg_train_loss)

    avg_loss_adv = avg_loss_adv / total_batches

    train_task2_accuracy = accuracy_score(train_task2_labels, train_task2_preds)

    precision_t2, recall_t2, f1_t2, _ = precision_recall_fscore_support(
        train_task2_labels, train_task2_preds, average='weighted', zero_division=1)

    train_task2_outputs = np.concatenate(train_task2_outputs, axis=0)

    train_task2_probs = torch.softmax(torch.from_numpy(train_task2_outputs), dim=1).cpu().numpy()

    train_task2_labels_binarized = label_binarize(train_task2_labels, classes=[0,1,2])
    auc_t2 = roc_auc_score(
        train_task2_labels_binarized,
        train_task2_probs,
        average='macro',
        multi_class='ovr'
    )

    print(f"Train epoch={epoch}, Avg Loss={avg_train_loss:.4f}, Fairness Loss={loss_fairness.item():.4f}, Adv Loss={avg_loss_adv:.4f}"
          f"Task2 accuracy={train_task2_accuracy:.4f}, "
          f"Precision={precision_t2:.4f}, Recall={recall_t2:.4f}, F1={f1_t2:.4f}, "
          f"AUC={auc_t2:.4f}")
    print(f"Train epoch={epoch}, Avg Loss={avg_train_loss:.4f}, Fairness Loss={loss_fairness.item():.4f}, Adv Loss={avg_loss_adv:.4f} "
          f"Task2 accuracy={train_task2_accuracy:.4f}, "
          f"Precision={precision_t2:.4f}, Recall={recall_t2:.4f}, F1={f1_t2:.4f}, "
          f"AUC={auc_t2:.4f}",
          file=writer)

    # =============== Testing =============== #
    model.eval()
    test_task2_labels, test_task2_preds = [], []
    test_task2_outputs = []
    test_loss = 0
    total_batches = len(test_loader)

    group_metrics = {
        'Gender': {
            0: {'pred_logits': [], 'label': []},  # Gender 0
            1: {'pred_logits': [], 'label': []}  # Gender 1
        },
    }

    with torch.no_grad():
        for index, (closed_pd_images, closed_t2w_images, open_pd_images, label2, gender) in enumerate(test_loader):
            label2 = label2.to(device)
            out_task2, _ = model(closed_pd_images, closed_t2w_images, open_pd_images)

            loss_task2 = criterion_task2(out_task2, label2)
            loss = loss_task2

            test_loss += loss.item()

            _, t2_preds_batch = torch.max(out_task2, 1)
            test_task2_labels.extend(label2.cpu().numpy())
            test_task2_preds.extend(t2_preds_batch.cpu().numpy())
            test_task2_outputs.append(out_task2.cpu().numpy())

    test_task2_outputs = np.concatenate(test_task2_outputs, axis=0)
    test_task2_accuracy = accuracy_score(test_task2_labels, test_task2_preds)

    acc_count_t2 = sum(np.array(test_task2_labels) == np.array(test_task2_preds))
    acc_lower_t2, acc_upper_t2 = wilson_confidence_interval(acc_count_t2, len(test_task2_labels))

    precision_t2, recall_t2, f1_t2, _ = precision_recall_fscore_support(
        test_task2_labels, test_task2_preds, average='weighted', zero_division=1)

    test_task2_probs = torch.softmax(torch.from_numpy(test_task2_outputs), dim=1).cpu().numpy()

    test_task2_labels_binarized = label_binarize(test_task2_labels, classes=[0,1,2])
    auc_t2 = roc_auc_score(
        test_task2_labels_binarized,
        test_task2_probs,
        average='macro',
        multi_class='ovr'
    )

    conf_matrix_t2 = confusion_matrix(test_task2_labels, test_task2_preds)

    avg_test_loss = test_loss / total_batches
    test_losses.append(avg_test_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Test Loss: {avg_test_loss:.4f}")

    print(f"Test epoch={epoch}, Task2 accuracy={test_task2_accuracy:.4f} "
          f"Precision={precision_t2:.4f}, Recall={recall_t2:.4f}, F1={f1_t2:.4f}, "
          f"AUC={auc_t2:.4f}")
    print(f"Test epoch={epoch}, Task2 Confusion Matrix:\n{conf_matrix_t2}", file=writer)

    print(f"Test epoch={epoch}, Task2 accuracy={test_task2_accuracy:.4f} "
          f"Precision={precision_t2:.4f}, Recall={recall_t2:.4f}, F1={f1_t2:.4f}, "
          f"AUC={auc_t2:.4f}", file=writer)

    plot_roc_curve(
        test_task2_labels_binarized,
        test_task2_probs,
        "Task2",
        3,
        epoch
    )

    if test_task2_accuracy > best_accuracy_task2:
        best_accuracy_task2 = test_task2_accuracy
        torch.save(model.state_dict(), os.path.join(checkpoint_folder, "dual-constraint-fairness_best.pth".format(fairness_method)))

    print(f"Best AUC so far - Task2: {best_auc_task2:.4f}")
    print(f"Best accuracy so far - Task2: {best_accuracy_task2:.4f}")
    print(f"Best AUC so far - Task2: {best_auc_task2:.4f}", file=writer)
    print(f"Best accuracy so far - Task2: {best_accuracy_task2:.4f}", file=writer)
    writer.flush()

writer.close()
