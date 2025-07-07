from jaxtyping import install_import_hook
import pickle

# Add runtime type checking to all imports.
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.provided_code import get_bunny, plot_point_cloud

if __name__ == "__main__":
    vertices, _ = get_bunny()
    plot_point_cloud(
        vertices,
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        zlim=(-2.0, 2.0),
    )
