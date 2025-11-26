# gnn_vuln_scanner/multilabel_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import hamming_loss, jaccard_score, classification_report
import numpy as np
from tqdm import tqdm
import random
import json
import pickle


class MultiLabelGNNTrainer:
    """多标签GNN训练器"""

    def __init__(self, model, device, num_classes=10, class_names=None):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]

        # 使用BCEWithLogitsLoss处理多标签分类
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for data in progress_bar:
            data = data.to(self.device)

            # 确保标签是float类型
            targets = data.y.float()

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 计算准确率（阈值为0.5）
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.numel()

            accuracy = 100. * correct_predictions / total_predictions
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.2f}%'
            })

        return total_loss / len(train_loader), accuracy

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validating"):
                data = data.to(self.device)
                targets = data.y.float()

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # 收集预测结果
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        if not all_predictions:
            return 0, 0, [], [], []

        avg_loss = total_loss / len(val_loader)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)

        # 计算各种指标
        hamming = hamming_loss(all_targets, all_predictions)
        try:
            jaccard = jaccard_score(all_targets, all_predictions, average='samples')
        except:
            jaccard = 0

        accuracy = 100. * (1 - hamming)

        return avg_loss, accuracy, all_predictions, all_targets, all_probabilities

    def train(self, train_loader, val_loader, epochs=30):
        best_acc = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 50)

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # 验证
            val_loss, val_acc, preds, targets, probs = self.validate(val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # 学习率调度
            self.scheduler.step()

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_multilabel_gnn_model.pth')
                print(f'✅ Best model saved with accuracy: {best_acc:.2f}%')

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_acc': best_acc
        }


def split_dataset(graph_data_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """划分数据集"""
    print("📊 划分数据集...")

    # 设置随机种子确保可重现性
    random.seed(42)
    random.shuffle(graph_data_list)

    total_size = len(graph_data_list)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_data = graph_data_list[:train_size]
    val_data = graph_data_list[train_size:train_size + val_size]
    test_data = graph_data_list[train_size + val_size:]

    print(f"   训练集: {len(train_data)} 个图")
    print(f"   验证集: {len(val_data)} 个图")
    print(f"   测试集: {len(test_data)} 个图")

    return train_data, val_data, test_data


def load_graph_dataset(dataset_file):
    """加载图数据集"""
    print(f"📂 加载图数据集: {dataset_file}")

    with open(dataset_file, 'rb') as f:
        graph_data_list = pickle.load(f)

    print(f"✅ 成功加载 {len(graph_data_list)} 个图")

    # 验证标签维度
    valid_data_list = []
    invalid_count = 0

    for data in graph_data_list:
        try:
            if hasattr(data, 'y') and len(data.y) == 10:  # 10个类别
                valid_data_list.append(data)
            else:
                invalid_count += 1
        except Exception as e:
            print(f"⚠️  移除无效数据: {e}")
            invalid_count += 1

    print(f"✅ 有效数据: {len(valid_data_list)} 个图 (移除无效数据: {invalid_count})")
    return valid_data_list


def analyze_multilabel_results(predictions, targets, class_names):
    """分析多标签分类结果"""
    print("\n📋 多标签分类详细报告:")

    # 每个类别的精确率、召回率、F1分数
    from sklearn.metrics import precision_recall_fscore_support

    # 计算每个类别的指标
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )

    print("📊 各类别性能指标:")
    print(f"{'类别':<20} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'支持数':<10}")
    print("-" * 70)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {int(support[i]):<10}")

    # 整体指标
    hamming = hamming_loss(targets, predictions)
    try:
        subset_accuracy = np.mean(np.all(predictions == targets, axis=1))
        jaccard = jaccard_score(targets, predictions, average='samples')
    except:
        subset_accuracy = 0
        jaccard = 0

    print(f"\n🎯 整体性能指标:")
    print(f"   Hamming Loss: {hamming:.4f}")
    print(f"   Subset Accuracy: {subset_accuracy:.4f}")
    print(f"   Jaccard Score: {jaccard:.4f}")


def main_train_multilabel_model():
    """训练多标签模型的主函数"""
    dataset_file = "data/processed_output/multilabel_graph_dataset.pkl"
    vuln_types = [
        'buffer_overflow', 'use_after_free', 'double_free', 'null_pointer',
        'integer_overflow', 'format_string', 'command_injection',
        'path_traversal', 'race_condition', 'memory_leak'
    ]

    try:
        print("🚀 开始多标签GNN漏洞检测模型训练...")
        print("=" * 60)

        # 加载图数据集
        graph_data_list = load_graph_dataset(dataset_file)

        if not graph_data_list:
            print("❌ 没有有效的训练数据!")
            return None, None

        # 划分数据集
        train_data, val_data, test_data = split_dataset(graph_data_list)

        # 创建数据加载器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {device}")

        # 检查是否有足够的数据
        if len(train_data) == 0 or len(val_data) == 0:
            print("❌ 训练集或验证集为空!")
            return None, None

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # 初始化模型
        model = MultiLabelVulnGNN(input_dim=13, hidden_dim=128, num_classes=10)
        print(f"🧠 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

        # 训练模型
        trainer = MultiLabelGNNTrainer(model, device, num_classes=10, class_names=vuln_types)
        results = trainer.train(train_loader, val_loader, epochs=30)

        print(f"\n🏆 训练完成!")
        print(f"📈 最佳验证准确率: {results['best_acc']:.2f}%")

        # 在测试集上评估
        print("\n🧪 测试集评估...")
        test_loss, test_acc, test_preds, test_targets, test_probs = trainer.validate(test_loader)
        print(f"🏁 测试集准确率: {test_acc:.2f}%")

        # 详细的分类报告
        analyze_multilabel_results(test_preds, test_targets, vuln_types)

        # 保存结果
        results['test_acc'] = test_acc
        with open('multilabel_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # 保存测试预测结果
        test_results = {
            'predictions': test_preds.tolist(),
            'targets': test_targets.tolist(),
            'probabilities': test_probs.tolist()
        }
        with open('multilabel_test_predictions.pkl', 'wb') as f:
            pickle.dump(test_results, f)

        print(f"\n💾 结果已保存!")
        print("   - 模型权重: best_multilabel_gnn_model.pth")
        print("   - 训练日志: multilabel_training_results.json")
        print("   - 测试结果: multilabel_test_predictions.pkl")

        return model, results

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main_train_multilabel_model()
