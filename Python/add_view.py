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
edges_df[['source', 'target']].to_csv(r'E:\OOP\Project_OOP\Python\link_process\edges_user.csv', index=False)
print("Dữ liệu các cạnh đã được lưu vào file edges_user.csv")

# --- Lưu danh sách các nút ---
# Lấy dữ liệu từ bảng node
nodes_table = p4c.get_table_columns(table='node', network=network_id)

# Chuyển đổi sang DataFrame
nodes_df = pd.DataFrame(nodes_table)

# Tạo cột 'node_id' từ cột 'name'
nodes_df['node_id'] = nodes_df['name']

# Lưu DataFrame vào file CSV với cột 'node_id' và các cột từ 0 đến 127
columns_to_keep = ['node_id'] + [str(i) for i in range(128)]
nodes_df[columns_to_keep].to_csv(r'E:\OOP\Project_OOP\Python\link_process\nodes_user.csv', index=False)
print("Dữ liệu các nút đã được lưu vào file nodes_user.csv")
import pandas as pd

# Đọc dữ liệu từ file link_nodesview.csv và nodes_user.csv
nodesview_df = pd.read_csv(r'E:\OOP\Project_OOP\Python\link_process\link_nodesview.csv')
nodesuser_df = pd.read_csv(r'E:\OOP\Project_OOP\Python\link_process\nodes_user.csv')

# Tìm các node_id có trong link_nodesview.csv mà không có trong nodes_user.csv
node_difference_df = nodesview_df[~nodesview_df['node_id'].isin(nodesuser_df['node_id'])]

# Bỏ cột 'label' trong kết quả
node_difference_df = node_difference_df.drop(columns=['label'])

# Lưu kết quả vào file node_difference.csv
node_difference_df.to_csv(r'E:\OOP\Project_OOP\Python\link_process\node_difference.csv', index=False)

print("Các node không có trong nodes_user.csv đã được lưu vào file node_difference.csv")
import py4cytoscape as p4c
import pandas as pd

# Đường dẫn đến file CSV
file_path = r"E:\OOP\Project_OOP\Python\link_process\node_difference.csv"

# Đọc dữ liệu từ file CSV
nodes_df = pd.read_csv(file_path)

# Kiểm tra cột node_id có tồn tại
if 'node_id' not in nodes_df.columns:
    raise ValueError("File CSV không chứa cột 'node_id'")

# Tạo DataFrame chứa các nút mới
nodes_data = pd.DataFrame({
    'name': nodes_df['node_id'].astype(str)  # Sử dụng 'name' thay vì 'id'
})

# Lấy đồ thị hiện tại trong Cytoscape
network_id = p4c.get_network_suid()

# Thêm các node mới vào đồ thị
p4c.add_cy_nodes(nodes_data['name'].tolist(), network=network_id)
print("Các node mới đã được thêm vào đồ thị.")

# Áp dụng thuật toán bố trí để sắp xếp các node
p4c.layout_network(layout_name='force-directed', network=network_id)
print("Các node đã được sắp xếp lại.")
import py4cytoscape as p4c
import pandas as pd

# Kết nối với Cytoscape
p4c.cytoscape_ping()

# Lấy SUID của mạng hiện tại
network_id = p4c.get_network_suid()
print("Network ID:", network_id)

# Lấy bảng các nút từ mạng hiện tại
nodes_table = p4c.get_table_columns(table='node', network=network_id)

# Chuyển dữ liệu thành DataFrame
nodes_df = pd.DataFrame(nodes_table)

# Kiểm tra các nút có trường 'label' rỗng
empty_label_nodes = nodes_df[nodes_df['label'].isnull()]

# Nếu có nút có label rỗng, đổi màu chúng thành xanh lá
if not empty_label_nodes.empty:
    node_ids = empty_label_nodes['SUID'].tolist()  # Chuyển đổi SUID của các nút có label rỗng thành danh sách
    # Cập nhật màu sắc của các nút này thành màu xanh lá
    p4c.set_node_property_bypass(node_ids, new_values='green', visual_property='NODE_FILL_COLOR', network=network_id)

    print(f"Đã đổi màu các nút có label rỗng thành xanh lá.")
else:
    print("Không có nút nào có label rỗng.")