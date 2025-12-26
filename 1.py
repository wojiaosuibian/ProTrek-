import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import time
import matplotlib.pyplot as plt  # [新增] 导入绘图库

# ==========================================
# 1. 环境与数据准备
# ==========================================
print("初始化环境...")
DEVICE = "cpu"  # 强制使用 CPU 以证明轻量性

# 真实数据：绿色荧光蛋白 (GFP) vs 红色荧光蛋白 (DsRed)
GFP_SEQ = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
RFP_SEQ = "MASSEDVIKEFMRFKVRMEGSMNGHEFEIEGEGEGRPYEGHNTVKLKVTKGGPLPFAWDILSPQFQYGSKVYVKHPADIPDYKKLSFPEGFKWERVMNFEDGGVVTVTQDSSLQDGCFIYKVKFIGVNFPSDGPVMQKKTMGWEASTERLYPRDGVLKGEIHKALKLKDGGHYLVEFKSIYMAKKPVQLPGYYYVDSKLDITSHNEDYTIVEQYERTEGRHHLFL"

# 训练数据构造
TRAIN_DATA = [
    (GFP_SEQ, "Green Fluorescent Protein", "Red Fluorescent Protein"),
    (RFP_SEQ, "Red Fluorescent Protein", "Green Fluorescent Protein")
]

# ==========================================
# 2. 定义改进版 ProTrek 模型架构
# ==========================================
class ProTrekGranular(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 蛋白质编码器 (选用 ESM-2 8M 微型版)
        print("加载蛋白质编码器 (ESM-2 8M)...")
        self.prot_name = "facebook/esm2_t6_8M_UR50D"
        self.prot_tokenizer = AutoTokenizer.from_pretrained(self.prot_name)
        self.prot_encoder = AutoModel.from_pretrained(self.prot_name)

        # 2. 文本编码器 (选用 MiniLM)
        print("加载文本编码器 (MiniLM)...")
        self.text_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_name)
        self.text_encoder = AutoModel.from_pretrained(self.text_name)

        # 3. 优化核心：粒度感知投影层
        self.prot_proj = nn.Linear(320, 128)
        self.text_proj = nn.Linear(384, 128)

        # 冻结预训练参数
        for param in self.prot_encoder.parameters(): param.requires_grad = False
        for param in self.text_encoder.parameters(): param.requires_grad = False

    def forward(self, seqs, texts):
        # 蛋白质编码
        p_inputs = self.prot_tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            p_out = self.prot_encoder(**p_inputs).last_hidden_state
        p_embed = p_out.mean(dim=1)
        p_vec = self.prot_proj(p_embed)

        # 文本编码
        t_inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            t_out = self.text_encoder(**t_inputs).last_hidden_state
        t_embed = t_out.mean(dim=1)
        t_vec = self.text_proj(t_embed)

        return torch.nn.functional.normalize(p_vec, p=2, dim=1), torch.nn.functional.normalize(t_vec, p=2, dim=1)

# ==========================================
# 3. 实例化与基准测试
# ==========================================
model = ProTrekGranular().to(DEVICE)
optimizer = optim.Adam(list(model.prot_proj.parameters()) + list(model.text_proj.parameters()), lr=1e-3)

def check_granularity(stage_name):
    print(f"\n--- {stage_name} 粒度测试 ---")
    model.eval()
    with torch.no_grad():
        p_vec, t_vec_green = model([GFP_SEQ], ["Green Fluorescent Protein"])
        _, t_vec_red = model([GFP_SEQ], ["Red Fluorescent Protein"])

        sim_green = torch.mm(p_vec, t_vec_green.T).item()
        sim_red = torch.mm(p_vec, t_vec_red.T).item()

        print(f"GFP 序列 <-> 'Green' 文本相似度: {sim_green:.4f}")
        print(f"GFP 序列 <-> 'Red'   文本相似度: {sim_red:.4f}")
        print(f"区分度 (Margin): {sim_green - sim_red:.4f}")
        return sim_green, sim_red

print("\n训练前基准测试：")
check_granularity("微调前 (Baseline)")

# ==========================================
# 4. 优化算法：对比学习微调 (Fine-tuning)
# ==========================================
print("\n⚙️ 开始高精度微调 (Simulating 'Granularity Optimization')...")
start_time = time.time()
model.train()

# [新增] 用于存储每个 Epoch 的 Loss，方便画图
loss_history = []

for epoch in range(20):
    total_loss = 0
    for seq, pos_text, neg_text in TRAIN_DATA:
        optimizer.zero_grad()

        # 计算正负样本对
        p_vec, pos_vec = model([seq], [pos_text])
        _, neg_vec = model([seq], [neg_text])

        # 损失函数
        pos_sim = torch.mm(p_vec, pos_vec.T)
        neg_sim = torch.mm(p_vec, neg_vec.T)
        loss = (1 - pos_sim) + torch.clamp(neg_sim + 0.2, min=0)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # [新增] 记录 Loss
    loss_history.append(total_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/20 | Loss: {total_loss:.4f}")

print(f"微调完成！耗时: {time.time() - start_time:.2f} 秒")

# ==========================================
# 5. 最终验证与绘图
# ==========================================
print("\n训练后效果验证：")
sim_g, sim_r = check_granularity("微调后 (Optimized)")

if sim_g > sim_r:
    print("\n模型现在能够有效区分 GFP 和 RFP 的细微语义差异。")
else:
    print("\n⚠还需要更多训练步骤。")

# [新增] 绘制 Loss 曲线图代码
print("\n正在生成 Loss 曲线图...")
plt.figure(figsize=(10, 6)) # 设置图片大小
plt.plot(range(1, 21), loss_history, marker='o', linestyle='-', color='b', label='Training Loss')

plt.title('Training Loss Curve (Hard Negative Mining)', fontsize=16) # 标题
plt.xlabel('Epochs', fontsize=14) # X轴标签
plt.ylabel('Total Loss', fontsize=14) # Y轴标签
plt.grid(True, linestyle='--', alpha=0.7) # 网格
plt.legend() # 图例
plt.tight_layout()

# 保存图片（可选）
plt.savefig('loss_curve.png')
print("图表已保存为 loss_curve.png")

# 显示图片
plt.show()