from pathlib import Path
from jaxtyping import install_import_hook

# Add runtime type checking to all imports.
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.rebuild import load_image, Finding_correspondences_and_generate_F, Fundamental, triangulate_points
    from src.rebuild import plot_model

# Put the path to your puzzle here.
IMAGE_PATH = Path("/home/jacksemual/PycharmProjects/6S980-hw1/data/sample_dataset")

if __name__ == "__main__":  
    images = load_image(IMAGE_PATH)
    F, inlier_matches, src_pts, dst_pts, all_kps, all_descs = Finding_correspondences_and_generate_F(images)
    P1,P2 = Fundamental(F)
    points = triangulate_points(P1, P2, inlier_matches, all_kps, src_pts, dst_pts)
    plot_model(points)