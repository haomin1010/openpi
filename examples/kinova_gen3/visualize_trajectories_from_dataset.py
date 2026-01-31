#!/usr/bin/env python3
"""
Visualize Kinova Gen3 end-effector trajectories from a dataset.

Load episodes stored in LeRobot Parquet format, compute forward kinematics,
and render 3D trajectories.

Usage examples:
    python visualize_trajectories_from_dataset.py
    python visualize_trajectories_from_dataset.py --max-episodes 5
    python visualize_trajectories_from_dataset.py --output-dir ./my_images
"""

import argparse
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd

from urdf_kinematics import URDFKinematics

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("VisualizeTrajectoriesDataset")


def load_episode_data(episode_path: Path) -> pd.DataFrame:
    """
    Load a single episode from Parquet.

    Args:
        episode_path: path to the episode file.

    Returns:
        DataFrame containing the episode steps, or None on error.
    """
    try:
        return pd.read_parquet(episode_path)
    except Exception as e:
        logger.error(f"Failed to load episode {episode_path}: {e}")
        return None


def compute_trajectory_stats(trajectory: np.ndarray) -> dict:
    """
    Compute trajectory statistics.

    Args:
        trajectory: (T, 3) trajectory array.

    Returns:
        dict with statistics.
    """
    if len(trajectory) < 2:
        return {}
    
    # Compute total path length.
    displacements = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    total_length = np.sum(displacements)
    
    # Compute straight-line distance from start to end.
    straight_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
    
    # Compute workspace range.
    workspace_min = trajectory.min(axis=0)
    workspace_max = trajectory.max(axis=0)
    workspace_range = workspace_max - workspace_min
    
    return {
        'num_points': len(trajectory),
        'total_length': total_length,
        'straight_distance': straight_distance,
        'tortuosity': total_length / straight_distance if straight_distance > 0 else 0,
        'workspace_min': workspace_min,
        'workspace_max': workspace_max,
        'workspace_range': workspace_range,
    }


def plot_single_trajectory(
    trajectory: np.ndarray,
    output_path: Path,
    title: str = "Kinova Gen3 End-Effector Trajectory",
    stats: dict = None
):
    """
    Plot a single 3D trajectory to a PNG file.

    Args:
        trajectory: (T, 3) array of EEF positions.
        output_path: destination path for the plot.
        title: plot title.
        stats: optional statistics for annotation.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))

    for i in range(len(trajectory) - 1):
        ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2],
                color=colors[i], linewidth=2, alpha=0.8)

    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
               color='green', s=200, marker='o', alpha=1.0, edgecolors='black', linewidths=2,
               label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
               color='red', s=200, marker='s', alpha=1.0, edgecolors='black', linewidths=2,
               label='End')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)

    if stats:
        stats_text = f"\nLength: {stats['total_length']:.3f}m | Points: {stats['num_points']}"
        ax.set_title(title + stats_text, fontsize=14, fontweight='bold')
    else:
        ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_end_effector_trajectory(states: np.ndarray, kinematics: URDFKinematics) -> np.ndarray:
    """
    Compute end-effector trajectory from joint states.

    Args:
        states: (T, 32) array where the first 7 dims are joint angles.
        kinematics: URDFKinematics instance.

    Returns:
        (T, 3) end-effector positions.
    """
    trajectories = []
    for state in states:
        joint_angles = state[:7]  # extract the first 7 joint angles
        _, eef_position = kinematics.compute_joint_positions(joint_angles)
        trajectories.append(eef_position)
    return np.array(trajectories)


def plot_trajectories_3d(
    trajectories: list[np.ndarray],
    episode_labels: list[str],
    output_path: Path,
    title: str = "Kinova Gen3 End-Effector Trajectories"
):
    """
    Plot multiple 3D trajectories in one figure.

    Args:
        trajectories: list of (T, 3) arrays.
        episode_labels: legend labels.
        output_path: destination path.
        title: plot title.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories)))

    for idx, (traj, label) in enumerate(zip(trajectories, episode_labels)):
        if len(traj) == 0:
            continue
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=colors[idx], linewidth=2, alpha=0.7, label=label)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                   color='green', s=100, marker='o', alpha=0.8)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                   color='red', s=100, marker='s', alpha=0.8)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved trajectory plot to: {output_path}")
    plt.close()


def main():
    """Main entrypoint."""
    script_dir = Path(__file__).parent
    default_data_dir = Path.home() / ".cache/huggingface/lerobot/kinova_gen3_dataset/data/chunk-000"
    default_urdf_path = script_dir / "GEN3_URDF_V12_with_dampint.urdf"
    default_output_dir = script_dir / "saved_images"

    parser = argparse.ArgumentParser(
        description="Visualize Kinova Gen3 dataset trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # visualize all episodes
  python visualize_trajectories_from_dataset.py

  # limit to first 5 episodes
  python visualize_trajectories_from_dataset.py --max-episodes 5

  # write to a custom directory
  python visualize_trajectories_from_dataset.py --output-dir ./my_images
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(default_data_dir),
        help=f'Dataset directory (default: {default_data_dir})'
    )
    parser.add_argument(
        '--urdf-path',
        type=str,
        default=str(default_urdf_path),
        help=f'URDF file path (default: {default_urdf_path})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(default_output_dir),
        help=f'Output directory (default: {default_output_dir})'
    )
    parser.add_argument(
        '--max-episodes',
        type=int,
        default=None,
        help='Maximum number of episodes to process (default: all)'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default='kinova_trajectories_3d.png',
        help='Output image filename (default: kinova_trajectories_3d.png)'
    )
    parser.add_argument(
        '--separate-plots',
        action='store_true',
        help='Create a separate plot for each episode'
    )
    parser.add_argument(
        '--show-stats',
        action='store_true',
        help='Display trajectory statistics'
    )
    
    args = parser.parse_args()
    
    # Validate data directory.
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Dataset directory not found: {data_dir}")
        sys.exit(1)
    
    # Validate URDF file.
    urdf_path = Path(args.urdf_path)
    if not urdf_path.exists():
        logger.error(f"URDF file not found: {urdf_path}")
        sys.exit(1)
    
    # Create output directory.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize forward kinematics.
    logger.info(f"Loading URDF: {urdf_path}")
    try:
        kinematics = URDFKinematics(urdf_path)
        logger.info(f"URDF loaded with {len(kinematics.actuated_joint_names)} actuated joints")
    except Exception as e:
        logger.error(f"Failed to load URDF: {e}")
        sys.exit(1)
    
    # Find episode files.
    episode_files = sorted(data_dir.glob("episode_*.parquet"))
    if not episode_files:
    logger.error(f"No episode files found: {data_dir}/episode_*.parquet")
    sys.exit(1)
    
    # Limit number of episodes if requested.
    if args.max_episodes is not None:
        episode_files = episode_files[:args.max_episodes]
    
    logger.info(f"Found {len(episode_files)} episode files")
    
    # Process each episode file.
    all_trajectories = []
    episode_labels = []
    all_stats = []
    
    for idx, episode_file in enumerate(episode_files):
        logger.info(f"Processing episode {idx + 1}/{len(episode_files)}: {episode_file.name}")
        
        # Load episode data.
        df = load_episode_data(episode_file)
        if df is None or len(df) == 0:
            logger.warning(f"Skipping empty episode: {episode_file.name}")
            continue
        
        # Extract state observations.
        states = np.stack(df['state'].values)  # (T, 32)
        logger.info(f"  Steps: {len(states)}")
        
        # Compute end-effector trajectory.
        try:
            trajectory = compute_end_effector_trajectory(states, kinematics)
            all_trajectories.append(trajectory)
            episode_labels.append(f"Episode {idx}")
            logger.info(f"  Trajectory computed, length: {len(trajectory)}")
            
            # Compute optional statistics.
            if args.show_stats:
                stats = compute_trajectory_stats(trajectory)
                all_stats.append(stats)
                logger.info(f"  Total length: {stats['total_length']:.3f}m")
                logger.info(
                    f"  Workspace range: X[{stats['workspace_min'][0]:.3f}, {stats['workspace_max'][0]:.3f}], "
                    f"Y[{stats['workspace_min'][1]:.3f}, {stats['workspace_max'][1]:.3f}], "
                    f"Z[{stats['workspace_min'][2]:.3f}, {stats['workspace_max'][2]:.3f}]"
                )
            
            # Save a separate plot for each episode.
            if args.separate_plots:
                single_output_path = output_dir / f"episode_{idx:03d}_trajectory.png"
                plot_single_trajectory(
                    trajectory,
                    single_output_path,
                title=f"Episode {idx} - End-Effector Trajectory",
                    stats=all_stats[-1] if args.show_stats else None
                )
                logger.info(f"  Saved separate image: {single_output_path}")
                
        except Exception as e:
            logger.error(f"  Failed to compute trajectory: {e}")
            continue
    
    if not all_trajectories:
    logger.error("No episodes were processed successfully")
    sys.exit(1)
        sys.exit(1)
    
    # Plot all trajectories.
    output_path = output_dir / args.output_filename
    logger.info(f"Plotting {len(all_trajectories)} trajectories...")
    try:
        plot_trajectories_3d(
            all_trajectories, 
            episode_labels, 
            output_path,
            title=f"Kinova Gen3 End-Effector Trajectories ({len(all_trajectories)} episodes)"
        )
    logger.info("Finished!")
    except Exception as e:
    logger.error(f"Failed to render plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
