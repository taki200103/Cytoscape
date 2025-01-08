import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sửa lại class GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, in_channels)  # out_channels = in_channels

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def map_edge_index_to_new_nodes(edge_index, existing_node_count, new_node_count):
    # Lọc bỏ các cạnh có chỉ số node vượt quá số node hiện có
    max_node_idx = existing_node_count + new_node_count - 1
    valid_edges = (edge_index[0] <= max_node_idx) & (edge_index[1] <= max_node_idx)
    edge_index = edge_index[:, valid_edges]
    
    # Ánh xạ node
    node_map = {i: i for i in range(existing_node_count)}
    node_map.update({existing_node_count + i: existing_node_count + i for i in range(new_node_count)})
    
    edge_index_mapped = edge_index.clone()
    edge_index_mapped[0] = torch.tensor([node_map.get(int(x), x) for x in edge_index[0]], dtype=torch.long)
    edge_index_mapped[1] = torch.tensor([node_map.get(int(x), x) for x in edge_index[1]], dtype=torch.long)
    
    return edge_index_mapped

# Sửa lại hàm load_and_train_model
def load_and_train_model(train_edges_file, train_nodes_file, new_nodes_file=None):
    # Đọc và xử lý dữ liệu
    edges_df = pd.read_csv(train_edges_file)
    nodes_df = pd.read_csv(train_nodes_file)
    
    # Đảm bảo các chỉ số trong edges_df không vượt quá số node
    max_node_id = len(nodes_df) - 1
    edges_df = edges_df[
        (edges_df['source'] <= max_node_id) & 
        (edges_df['target'] <= max_node_id)
    ]

    # Đọc thêm node mới nếu có
    if new_nodes_file:
        new_nodes_df = pd.read_csv(new_nodes_file)
        new_node_features = torch.tensor(new_nodes_df.iloc[:, 1:].values, dtype=torch.float)
    else:
        new_node_features = torch.tensor([], dtype=torch.float)

    # Số lượng node trong đồ thị hiện tại
    existing_node_count = len(nodes_df)
    all_node_features = torch.tensor(nodes_df.iloc[:, 1:].values, dtype=torch.float)

    # Ánh xạ lại chỉ số các cạnh
    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)
    edge_index = map_edge_index_to_new_nodes(edge_index, existing_node_count, new_nodes_df.shape[0] if new_nodes_file else 0)

    # Ghép các node mới vào node features
    all_node_features = torch.cat([all_node_features, new_node_features], dim=0)

    # Tạo đối tượng dữ liệu cho PyTorch Geometric
    data = Data(x=all_node_features, edge_index=edge_index)

    # Khởi tạo model với out_channels = in_channels
    model = GCN(in_channels=all_node_features.shape[1], hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Huấn luyện mô hình
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Lưu mô hình
    torch.save(model.state_dict(), "link_predict.pth")

    return model, data

# Sửa lại hàm predict_with_trained_model 
def predict_with_trained_model(model_path, data, new_nodes_file):
    # Tải mô hình đã huấn luyện
    model = GCN(in_channels=data.x.shape[1], hidden_channels=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Đọc node mới và đảm bảo số features giống nhau
    new_nodes_df = pd.read_csv(new_nodes_file)
    expected_features = data.x.shape[1]  # Số features của nodes hiện tại
    
    # Chỉ lấy số cột features giống với nodes hiện tại
    new_node_features = torch.tensor(new_nodes_df.iloc[:, 1:expected_features+1].values, dtype=torch.float)

    # Ghép các node mới vào node_features hiện tại
    all_node_features = torch.cat([data.x, new_node_features], dim=0)

    # Dự đoán embedding cho tất cả các node, bao gồm node mới
    edge_index = data.edge_index
    with torch.no_grad():
        output_embeddings = model(all_node_features, edge_index)

    return output_embeddings, all_node_features

def compute_similarity_and_create_edges(output_embeddings, existing_node_count, edges_df, threshold=0.5, max_edges=2):
    embeddings_np = output_embeddings if isinstance(output_embeddings, np.ndarray) else output_embeddings.numpy()
    
    new_nodes_embeddings = embeddings_np[existing_node_count:]
    existing_nodes_embeddings = embeddings_np[:existing_node_count]
    
    similarities = cosine_similarity(new_nodes_embeddings, existing_nodes_embeddings)
    
    new_edges = []
    for i in range(len(new_nodes_embeddings)):
        new_node_idx = i + existing_node_count
        # Get top 2 most similar existing nodes
        top_similar_indices = np.argsort(similarities[i])[-max_edges:]
        
        for target_idx in top_similar_indices:
            if similarities[i][target_idx] > threshold:
                new_edges.append([new_node_idx, int(target_idx)])
    
    # Combine old and new edges
    old_edges = edges_df.values.tolist()
    all_edges = old_edges + new_edges
    
    return all_edges

# Main Code
if __name__ == "__main__":
    # Đọc dữ liệu ban đầu và huấn luyện mô hình
    model, data = load_and_train_model("E:/OOP/Project_OOP/Python/link_process/edges_user.csv", 
                                       "E:/OOP/Project_OOP/Python/link_process/nodes_user.csv")
    
    # Dự đoán với node mới
    output_embeddings, all_node_features = predict_with_trained_model("link_predict.pth", data, "E:/OOP/Project_OOP/Python/link_process/node_difference.csv")
    
    # Lưu embeddings và các node mới vào CSV
    output_node_ids = torch.arange(all_node_features.shape[0]).numpy()
    output_embeddings = output_embeddings.numpy()

    # Lưu node embeddings vào CSV
    node_embeddings_df = pd.DataFrame(output_embeddings)
    node_embeddings_df.insert(0, 'node_id', output_node_ids)
    node_embeddings_df.to_csv(r'E:\OOP\Project_OOP\Python\link_process\all_embeddings.csv', index=False)

    # Read original edges
    edges_df = pd.read_csv("E:/OOP/Project_OOP/Python/link_process/edges_user.csv")
    
    # Get all edges including predictions
    all_edges = compute_similarity_and_create_edges(
        output_embeddings,
        len(data.x),
        edges_df,
        threshold=0.5
    )
    
    # Save all edges
    all_edges_df = pd.DataFrame(all_edges, columns=['source', 'target'])
    all_edges_df.to_csv(r'E:\OOP\Project_OOP\Python\link_process\all_edges.csv', index=False)
import py4cytoscape as p4c
import pandas as pd

# Kết nối với Cytoscape
p4c.cytoscape_ping()

# Đọc dữ liệu từ tệp CSV (source, target)
edges_df = pd.read_csv(r'E:\OOP\Project_OOP\Python\link_process\all_edges.csv')

# Kiểm tra dữ liệu đã đọc
print(edges_df.head())

# Lấy danh sách các nút từ các cột 'source' và 'target'
nodes_list = pd.concat([edges_df['source'], edges_df['target']]).unique()

# Tạo DataFrame cho các nút
nodes_df = pd.DataFrame(nodes_list, columns=['id'])

# Thêm cột 'interaction' nếu chưa có
if 'interaction' not in edges_df.columns:
    edges_df['interaction'] = 'interacts'

# Chuyển đổi các giá trị thành chuỗi
edges_df['source'] = edges_df['source'].astype(str)
edges_df['target'] = edges_df['target'].astype(str)
nodes_df['id'] = nodes_df['id'].astype(str)

# Tạo mạng từ các DataFrame của nút và cạnh
network = p4c.create_network_from_data_frames(nodes_df, edges_df, title='Link Network', collection='My Collection')


# Kiểm tra mạng đã được tạo thành công
print("Network created successfully!")
import py4cytoscape as cy
import os

# Đường dẫn tới tệp CSV
file_path = r'E:\OOP\Project_OOP\Python\link_process\all_embeddings.csv'

# Kiểm tra xem tệp có tồn tại không
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # Tải dữ liệu bảng vào Cytoscape
    try:
        cy.load_table_data_from_file(file_path)
        # Kiểm tra các cột trong bảng dữ liệu đã được tải vào Cytoscape
        print(cy.get_table_columns())
    except Exception as e:
        print(f"Error loading table data: {e}")