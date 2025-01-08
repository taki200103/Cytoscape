import py4cytoscape as p4c
import pandas as pd

# Kết nối với Cytoscape và lấy SUID của mạng hiện tại
network_id = p4c.get_network_suid()
print("Network ID:", network_id)

# --- Lưu danh sách các cạnh ---
# Lấy dữ liệu từ bảng edge
edges_table = p4c.get_table_columns(table='edge', network=network_id)

# Chuyển đổi sang DataFrame
edges_df = pd.DataFrame(edges_table)

# Tách cột 'shared name' thành 'source' và 'target'
edges_df[['source', 'target']] = edges_df['shared name'].str.extract(r'(\d+) \(interacts with\) (\d+)')

# Chuyển đổi các cột 'source' và 'target' sang kiểu số nguyên
edges_df['source'] = edges_df['source'].astype(int)
edges_df['target'] = edges_df['target'].astype(int)

# Lưu DataFrame vào file CSV với các cột 'source' và 'target'
edges_df[['source', 'target']].to_csv(r'E:\OOP\Project_OOP\Python\process\edges_view.csv', index=False)
print("Dữ liệu các cạnh đã được lưu vào file edges_view.csv")

# --- Lưu danh sách các nút ---
# Lấy dữ liệu từ bảng node
nodes_table = p4c.get_table_columns(table='node', network=network_id)

# Chuyển đổi sang DataFrame
nodes_df = pd.DataFrame(nodes_table)

# Tạo cột 'node_id' từ cột 'name'
nodes_df['node_id'] = nodes_df['name']

# Lưu DataFrame vào file CSV với cột 'node_id' và các cột từ 0 đến 127
columns_to_keep = ['node_id'] + [str(i) for i in range(128)]
nodes_df[columns_to_keep].to_csv(r'E:\OOP\Project_OOP\Python\process\nodes_view.csv', index=False)
print("Dữ liệu các nút đã được lưu vào file nodes_view.csv")
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import os

# 1. Định nghĩa mô hình GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # Sử dụng softmax cho classification

# 2. Load dữ liệu từ file CSV
def load_data(nodes_file, edges_file):
    # Cập nhật đường dẫn đúng tới tệp dữ liệu trên máy tính của bạn
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)

    # Thêm cột 'label' với giá trị mặc định là 0
    if 'label' not in nodes.columns:
        nodes['label'] = 0

    x = torch.tensor(nodes.iloc[:, 2:].values, dtype=torch.float)
    y = torch.tensor(nodes['label'].values, dtype=torch.long)
    edge_index = torch.tensor(edges[['source', 'target']].values.T, dtype=torch.long)

    num_nodes = x.size(0)
    val_mask = torch.ones(num_nodes, dtype=torch.bool)  # Mask cho tập validation

    data = Data(x=x, edge_index=edge_index, y=y, val_mask=val_mask)
    return data

# 3. Load mô hình đã huấn luyện với tham số strict=False để bỏ qua các tham số không khớp
def load_model(model_path, in_channels, hidden_channels, out_channels):
    model = GCN(in_channels, hidden_channels, out_channels)
    
    # Load state dict từ file và bỏ qua các tham số không khớp
    state_dict = torch.load(model_path)
    
    # Kiểm tra các tham số không có trong state_dict và thay thế bằng 0 nếu cần thiết
    for param_name, param_value in state_dict.items():
        if param_name not in model.state_dict():
            print(f"Warning: Parameter {param_name} not in model. Setting to zero.")
            state_dict[param_name] = torch.zeros_like(param_value)
    
    # Cập nhật mô hình với state dict đã được xử lý
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # Chế độ đánh giá
    return model

# 4. Tính toán node embeddings cho tập validation và dự đoán label
def compute_embeddings(model, data, save_dir):
    model.eval()
    with torch.no_grad():
        # Lấy đầu ra từ mô hình
        out = model(data.x, data.edge_index)
        node_embeddings = out[data.val_mask]  # Lấy embeddings của các node trong tập validation

        # Dự đoán label từ đầu ra của mô hình
        _, predicted_labels = out.max(dim=1)
        predicted_labels = predicted_labels[data.val_mask]  # Dự đoán label cho các node trong tập validation

    # Lưu node embeddings và labels ra file CSV
    node_ids = data.val_mask.nonzero(as_tuple=True)[0].cpu().numpy()  # Lấy node_id từ val_mask
    node_labels = predicted_labels.cpu().numpy()  # Dùng labels dự đoán từ mô hình
    
    # Tạo DataFrame để lưu trữ kết quả
    embeddings_df = pd.DataFrame(node_embeddings.cpu().numpy())
    embeddings_df['id'] = node_ids  # Đổi tên cột từ node_id thành id
    embeddings_df['label'] = node_labels
    
    # Đặt id và label ở vị trí đầu tiên
    columns = ['id', 'label'] + [col for col in embeddings_df.columns if col not in ['id', 'label']]
    embeddings_df = embeddings_df[columns]
    
    # Tạo thư mục lưu trữ nếu chưa có
    os.makedirs(save_dir, exist_ok=True)
    
    # Đường dẫn đầy đủ đến thư mục arixv
    file_path = os.path.join(save_dir, 'node_embeddings.csv')
    
    # Lưu vào file CSV
    embeddings_df.to_csv(file_path, index=False)
    print(f"Node embeddings and predicted labels saved to {file_path}")

# 5. Chạy tính toán embeddings trên tập validation
if __name__ == "__main__":
    val_nodes_file = r'E:\OOP\Project_OOP\Python\process\nodes_view.csv'
    val_edges_file = r'E:\OOP\Project_OOP\Python\process\edges_view.csv'
    model_path = r"E:\OOP\Project_OOP\Python\gcn_model.pth"  # Đường dẫn tới mô hình đã lưu
    save_dir = r"E:\OOP\Project_OOP\Python\process"  # Thư mục lưu node embeddings

    # Load dữ liệu validation
    data = load_data(val_nodes_file, val_edges_file)
    in_channels = data.x.shape[1]
    hidden_channels = 64
    out_channels = 40  # Đảm bảo out_channels đúng với mô hình đã huấn luyện , bởi vì ở hiện tại mô hình có ít label hơn 

    # Load mô hình đã huấn luyện từ file
    model = load_model(model_path, in_channels, hidden_channels, out_channels)

    # Tính toán và lưu node embeddings cho tập validation và dự đoán label
    compute_embeddings(model, data, save_dir)
import py4cytoscape as p4c
import pandas as pd

# Kết nối với Cytoscape
p4c.cytoscape_ping()

# Đọc dữ liệu từ tệp CSV (source, target)
edges_df = pd.read_csv(r'E:\OOP\Project_OOP\Python\process\edges_view.csv')

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
network = p4c.create_network_from_data_frames(nodes_df, edges_df, title='Embedding Network', collection='My Collection')


# Kiểm tra mạng đã được tạo thành công
print("Network created successfully!")
import py4cytoscape as cy
import os

# Đường dẫn tới tệp CSV
file_path = r'E:\OOP\Project_OOP\Python\process\node_embeddings.csv'

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