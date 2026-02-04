from pathlib import Path


def test_slurm_requests_4_gpus():
    script_path = Path(__file__).resolve().parents[1] / "slurm" / "train_dinowm.sh"
    content = script_path.read_text()

    assert "#SBATCH --gres=gpu:4" in content
    assert "#SBATCH --ntasks-per-node=4" in content
    assert "srun --ntasks=4 --gpus-per-task=1" in content
