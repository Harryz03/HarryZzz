import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision import models
import wikipediaapi
import requests
import time

class ERICFND(nn.Module):
    def __init__(self, hidden_dim=768):
        super(ERICFND, self).__init__()

        self.hidden_dim = hidden_dim

        # 文本编码器
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.text_fc = nn.Linear(hidden_dim, hidden_dim)

        # 图像编码器
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_fc = nn.Linear(resnet.fc.in_features, hidden_dim)

        # 实体编码层
        self.entity_fc = nn.Linear(hidden_dim, hidden_dim)
        self.attn_weight = nn.Linear(hidden_dim, 1)

        # 跨模态对比映射
        self.shared_fc_text = nn.Linear(hidden_dim, hidden_dim)
        self.shared_fc_img = nn.Linear(hidden_dim, hidden_dim)

        # 跨模态语义交互模块
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.interaction_fc = nn.Sequential(
            nn.Linear(hidden_dim * hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 自适应融合模块
        self.fusion_fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.gate_fc = nn.Linear(hidden_dim, 3)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )

        # 初始化维基百科API
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='ERICFND_Model/1.0 (harryzas13@gmail.com)'
        )



    def get_entity_descriptions(self, entity_list, sleep_time=0.5, retry=3):
        descriptions = []
        for entity in entity_list:
            for _ in range(retry):
                try:
                    page = self.wiki.page(entity)
                    if page.exists():
                        descriptions.append(page.summary)
                    else:
                        descriptions.append("")
                    break  # 成功就跳出重试
                except requests.exceptions.JSONDecodeError:
                    print(f"[跳过] 无法解析实体 {entity} 的 Wikipedia 响应。重试中...")
                    time.sleep(1)
                except Exception as e:
                    print(f"[异常] 获取 {entity} 摘要失败: {e}")
                    time.sleep(1)
            else:
                descriptions.append("")  # 重试失败也补空
            time.sleep(sleep_time)  # 控制访问频率
        return descriptions

    def encode_entity_descriptions(self, entity_list):
        """
        输入实体名称列表，调用维基百科API取摘要，
        用BERT编码，返回tensor(batch_size, num_entity, hidden_dim)
        """
        descriptions = self.get_entity_descriptions(entity_list)
        inputs = self.tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True, max_length=256)

        # 批量编码实体描述
        with torch.no_grad():
            outputs = self.bert(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            embeddings = outputs.pooler_output  # (num_entity, hidden_dim)

        return embeddings.unsqueeze(0)  # 加batch维度，变成(1, num_entity, hidden_dim)

    def forward(self, input_ids, attention_mask, image_tensor, entity_names):
        """
        entity_names: list[str], 实体名称列表（长度可变）
        """
        # 文本编码
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        r_t = self.text_fc(text_out.pooler_output)

        # 图像编码
        r_v = self.image_encoder(image_tensor)
        r_v = r_v.view(r_v.size(0), -1)  # 保持 [batch, feature_dim]
        r_v = self.image_fc(r_v)  # [batch, hidden_dim]

        # 动态编码实体描述
        entity_embeddings = self.encode_entity_descriptions(entity_names)  # (1, num_entity, hidden_dim)
        entity_embeddings = entity_embeddings.to(r_t.device)

        # 实体注意力融合
        attn_scores = torch.matmul(entity_embeddings, r_t.unsqueeze(-1)).squeeze(-1)  # (1, num_entity)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (1, num_entity, 1)
        r_d = torch.sum(attn_weights * entity_embeddings, dim=1)
        r_t_enhanced = r_t + self.entity_fc(r_d)

        # 跨模态映射
        e_t = self.shared_fc_text(r_t_enhanced)
        e_v = self.shared_fc_img(r_v)

        # 跨模态交互注意力
        if e_v.dim() == 1:
            e_v = e_v.unsqueeze(0)  # 变成 [1, hidden_dim]
        attn_output, _ = self.cross_attention(e_t.unsqueeze(1), e_v.unsqueeze(1), e_v.unsqueeze(1))
        cross_tensor = torch.bmm(attn_output.transpose(1, 2), attn_output)
        rf = self.interaction_fc(cross_tensor.view(cross_tensor.size(0), -1))

        # 自适应融合
        combined = torch.cat([r_t_enhanced, r_v, rf], dim=1)
        fusion_repr = self.fusion_fc(combined)
        gate_weights = torch.sigmoid(self.gate_fc(fusion_repr))
        final_repr = gate_weights[:, 0:1] * r_t_enhanced + gate_weights[:, 1:2] * r_v + gate_weights[:, 2:3] * rf

        # 分类输出
        output = self.classifier(final_repr)
        return output, e_t, e_v
