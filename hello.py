import numpy as np
import matplotlib.pyplot as plt
import argparse

def make_heart_curve(samples: int = 1000):
    t = np.linspace(0, 2 * np.pi, samples)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return x, y

def plot_heart(save_path: str | None = None, samples: int = 1000):
    x, y = make_heart_curve(samples)

    plt.figure(figsize=(6, 6))
    plt.fill(x, y, color="red", edgecolor="darkred")
    plt.plot(x, y, color="darkred", linewidth=2)
    plt.axis("equal")
    plt.axis("off")
    plt.title("Love", fontsize=20, color="darkred")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Draw a heart curve with matplotlib.")
    parser.add_argument("--save", help="Save output image to a file path instead of showing a window.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of points for the curve.")
    args = parser.parse_args()

    plot_heart(save_path=args.save, samples=args.samples)

if __name__ == "__main__":
    main()
