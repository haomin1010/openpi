#!/usr/bin/env python3
"""
Visualize Kinova Gen3 end-effector trajectories from control logs.

Load JSONL logs, compute the end-effector path via forward kinematics,
and generate 3D trajectory plots.

Usage:
    python visualize_trajectories_from_logs.py
    python visualize_trajectories_from_logs.py --log-file control_log_2026_01_30_16-51-00.jsonl
    python visualize_trajectories_from_logs.py --logs-dir ../../logs --max-logs 5
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from urdf_kinematics import URDFKinematics

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("VisualizeFromLogs")


def load_log_file(log_path: Path) -> list[dict]:
    """
    Load a single JSONL log file.

    Args:
        log_path: path to the log file.

    Returns:
        list of parsed log entries.
    """
    try:
        entries = []
        with open(log_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"  Skipping invalid JSON (line {line_num}): {e}")
                    continue
        return entries
    except Exception as e:
        logger.error(f"Failed to load log file {log_path}: {e}")
        return []


def extract_trajectory_from_log(log_entries: list[dict], kinematics: URDFKinematics,
                                use_state: bool = True) -> np.ndarray:
    """
    Extract the end-effector trajectory from log entries.

    Args:
        log_entries: parsed log entries.
        kinematics: URDFKinematics instance.
        use_state: use state.joint_position when True; otherwise use action.joint_position.

    Returns:
        (T, 3) numpy array of end-effector positions.
    """
    trajectories = []
    field_name = 'state' if use_state else 'action'
    
    for entry in log_entries:
        try:
            if field_name not in entry:
                continue
            
            joint_positions = entry[field_name].get('joint_position')
            if joint_positions is None or len(joint_positions) < 7:
                continue
            
            # Extract the first 7 joint angles.
            joint_angles = joint_positions[:7]
            _, eef_position = kinematics.compute_joint_positions(joint_angles)
            trajectories.append(eef_position)
        except Exception as e:
            logger.debug(f"Skipping entry: {e}")
            continue
    
    return np.array(trajectories) if trajectories else np.array([]).reshape(0, 3)


def compute_trajectory_stats(trajectory: np.ndarray) -> dict:
    """
    Compute statistics for a trajectory.

    Args:
        trajectory: (T, 3) array of positions.

    Returns:
        dict containing trajectory metrics.
    """
    if len(trajectory) < 2:
        return {}
    
    # Compute the total path length.
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
    title: str = "Kinova Gen3 End-Effector Trajectory (Log)",
    stats: dict = None
):
    """
    Plot a single trajectory with start/end markers.

    Args:
        trajectory: (T, 3) array of positions.
        output_path: path for saving the image.
        title: plot title.
        stats: optional statistics for annotating the title.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use gradient colors to show temporal progression.
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
    
    # Plot segmented trajectory lines.
    for i in range(len(trajectory) - 1):
        ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2],
                color=colors[i], linewidth=2, alpha=0.8)
    
    # Mark the start point (green circle).
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
              color='green', s=200, marker='o', alpha=1.0, edgecolors='black', linewidths=2,
              label='Start')
    
    # Mark the end point (red square).
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
              color='red', s=200, marker='s', alpha=1.0, edgecolors='black', linewidths=2,
              label='End')
    
    # Set axis labels.
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    
    # Append stats to the title if available.
    if stats:
        stats_text = f"\nLength: {stats['total_length']:.3f}m | Points: {stats['num_points']}"
        ax.set_title(title + stats_text, fontsize=14, fontweight='bold')
    else:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend.
    ax.legend(loc='upper left', fontsize=10)
    
    # Enable grid.
    ax.grid(True, alpha=0.3)
    
    # Set view angle.
    ax.view_init(elev=20, azim=45)
    
    # Save the figure.
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_trajectories_3d(
    trajectories: list[np.ndarray],
    labels: list[str],
    output_path: Path,
    title: str = "Kinova Gen3 End-Effector Trajectories (Logs)"
):
    """
    Plot multiple trajectories in one figure.

    Args:
        trajectories: list of (T, 3) arrays.
        labels: legend labels for each trajectory.
        output_path: destination path.
        title: plot title.
    """
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use different colors for each trajectory.
    colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories)))
    
    for idx, (traj, label) in enumerate(zip(trajectories, labels)):
        if len(traj) == 0:
            continue
        
        # Plot the trajectory line.
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=colors[idx], linewidth=2, alpha=0.7, label=label)
        
        # Mark the start point.
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                  color='green', s=100, marker='o', alpha=0.8)
        
        # Mark the end point.
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                  color='red', s=100, marker='s', alpha=0.8)
    
    # Set axis labels.
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend.
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    
    # Enable grid.
    ax.grid(True, alpha=0.3)
    
    # Set view angle.
    ax.view_init(elev=20, azim=45)
    
    # Save the figure.
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved trajectory plot to: {output_path}")
    plt.close()


def main():
    """Main entry point."""
    # Locate script directory.
    script_dir = Path(__file__).parent
    default_logs_dir = script_dir / "../../logs"
    default_urdf_path = script_dir / "GEN3_URDF_V12_with_dampint.urdf"
    default_output_dir = script_dir / "saved_images"
    
    parser = argparse.ArgumentParser(
        description="Visualize Kinova Gen3 end-effector trajectories from logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # visualize all logs under the logs directory
  python visualize_trajectories_from_logs.py

  # process a single log file
  python visualize_trajectories_from_logs.py --log-file ../../logs/control_log_2026_01_30_16-51-00.jsonl

  # limit to the first 5 log files
  python visualize_trajectories_from_logs.py --max-logs 5

  # generate per-log plots and show stats
  python visualize_trajectories_from_logs.py --max-logs 3 --separate-plots --show-stats
        """
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to a single log file (process only this file when set)'
    )
    parser.add_argument(
        '--logs-dir',
        type=str,
        default=str(default_logs_dir),
        help=f'Logs directory (default: {default_logs_dir})'
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
        '--max-logs',
        type=int,
        default=None,
        help='Maximum number of log files to process (default: all)'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default='kinova_trajectories_from_logs.png',
        help='Output image filename (default: kinova_trajectories_from_logs.png)'
    )
    parser.add_argument(
        '--separate-plots',
        action='store_true',
        help='Generate a separate image for each log file'
    )
    parser.add_argument(
        '--show-stats',
        action='store_true',
        help='Display trajectory statistics'
    )
    parser.add_argument(
        '--use-action',
        action='store_true',
        help='Use action.joint_position instead of state.joint_position'
    )
    
    args = parser.parse_args()
    
    # Validate URDF path.
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
    
    # Collect log files.
    log_files = []
    if args.log_file:
        # Handle single log file.
        log_file = Path(args.log_file)
        if not log_file.exists():
            logger.error(f"Log file not found: {log_file}")
            sys.exit(1)
        log_files = [log_file]
    else:
        # Look up all logs inside directory.
        logs_dir = Path(args.logs_dir)
        if not logs_dir.exists():
            logger.error(f"Logs directory not found: {logs_dir}")
            sys.exit(1)
        log_files = sorted(logs_dir.glob("control_log_*.jsonl"))
        if not log_files:
            logger.error(f"No log files found: {logs_dir}/control_log_*.jsonl")
            sys.exit(1)
    
    # Limit the number of files if requested.
    if args.max_logs is not None:
        log_files = log_files[:args.max_logs]
    
    logger.info(f"Found {len(log_files)} log files")
    
    # Process each log file.
    all_trajectories = []
    log_labels = []
    all_stats = []
    
    use_field = 'action' if args.use_action else 'state'
    logger.info(f"Using field: {use_field}.joint_position")
    
    for idx, log_file in enumerate(log_files):
        logger.info(f"Processing log {idx + 1}/{len(log_files)}: {log_file.name}")
        
        # Load log entries.
        log_entries = load_log_file(log_file)
        if not log_entries:
            logger.warning(f"Skipping empty log: {log_file.name}")
            continue
        
        logger.info(f"  Entries: {len(log_entries)}")
        
        # Compute end-effector trajectory.
        try:
            trajectory = extract_trajectory_from_log(
                log_entries, kinematics, use_state=not args.use_action
            )
            
            if len(trajectory) == 0:
                logger.warning(f"  Unable to extract a valid trajectory")
                continue
            
            all_trajectories.append(trajectory)
            log_labels.append(log_file.stem)  # use filename (without extension) as label
            logger.info(f"  Trajectory computed, points: {len(trajectory)}")
            
            # Collect statistics.
            if args.show_stats:
                stats = compute_trajectory_stats(trajectory)
                all_stats.append(stats)
                logger.info(f"  Total length: {stats['total_length']:.3f}m")
                logger.info(
                    f"  Workspace range: X[{stats['workspace_min'][0]:.3f}, {stats['workspace_max'][0]:.3f}], "
                    f"Y[{stats['workspace_min'][1]:.3f}, {stats['workspace_max'][1]:.3f}], "
                    f"Z[{stats['workspace_min'][2]:.3f}, {stats['workspace_max'][2]:.3f}]"
                )
            
            # Save separate images if requested.
            if args.separate_plots:
                # Use timestamped filename.
                single_output_path = output_dir / f"{log_file.stem}_trajectory.png"
                plot_single_trajectory(
                    trajectory,
                    single_output_path,
                    title=f"{log_file.stem}",
                    stats=all_stats[-1] if args.show_stats else None
                )
                logger.info(f"  Saved separate image: {single_output_path.name}")
                
        except Exception as e:
            logger.error(f"  Processing failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_trajectories:
        logger.error("No trajectories were successfully processed")
        sys.exit(1)
    
    # Plot all trajectories together.
    output_path = output_dir / args.output_filename
    logger.info(f"Plotting {len(all_trajectories)} trajectories...")
    try:
        plot_trajectories_3d(
            all_trajectories, 
            log_labels, 
            output_path,
            title=f"Kinova Gen3 End-Effector Trajectories (Logs, {len(all_trajectories)} paths)"
        )
        logger.info("Done!")
    except Exception as e:
        logger.error(f"Failed to render plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
