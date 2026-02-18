import open3d as o3d
import sys

ply_path = sys.argv[1] if len(sys.argv) > 1 else "output/dolls_sgm.ply"
point_cloud = o3d.io.read_point_cloud(ply_path)

# Display the point cloud
o3d.visualization.draw_geometries([point_cloud])