import contextlib
import io
import json
from PIL import Image
from modern.cli import main
from modern.experiment_tracker import log_experiment, best_experiment, read_experiments


def run_cli(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(list(argv))
    return code, out.getvalue()


def test_metrics_command(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("y_true,y_pred\ncat,cat\ncat,dog\ndog,dog\ndog,dog\n")
    code, out = run_cli("metrics", str(f))
    data = json.loads(out)
    assert code == 0 and data["accuracy"] == 0.75 and data["sample_count"] == 4


def test_metrics_command_reports_missing_column(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("a,b\n1,2\n")
    code, out = run_cli("metrics", str(f))
    assert code == 2 and out == ""


def test_profile_and_duplicates_commands(tmp_path):
    csv_file = tmp_path / "d.csv"
    csv_file.write_text("x,y\n1,a\n2,b\n")
    code, out = run_cli("profile", str(csv_file))
    assert code == 0 and json.loads(out)["row_count"] == 2
    for label in ("a", "b"):
        (tmp_path / "imgs" / label).mkdir(parents=True)
        Image.new("RGB", (4, 4), (5, 5, 5)).save(tmp_path / "imgs" / label / "x.png")
    code, out = run_cli("duplicates", str(tmp_path / "imgs"))
    assert code == 0 and json.loads(out)["cross_class_groups"] == 1


def test_split_command_omits_full_manifest_from_stdout(tmp_path):
    for label, base in (("a", 0), ("b", 100)):
        (tmp_path / "d" / label).mkdir(parents=True)
        for i in range(6):
            Image.new("RGB", (4, 4), (base + i * 9, 1, 2)).save(tmp_path / "d" / label / f"{i}.png")
    out_file = tmp_path / "m.json"
    code, out = run_cli("split", str(tmp_path / "d"), "--output", str(out_file), "--seed", "5")
    data = json.loads(out)
    assert code == 0 and "manifest" not in data and data["sample_count"] == 12
    assert len(json.loads(out_file.read_text())["manifest"]) == 12


def test_experiments_best_run(tmp_path):
    log = str(tmp_path / "runs.jsonl")
    log_experiment("a", {"accuracy": 0.80, "loss": 0.5}, path=log)
    log_experiment("b", {"accuracy": 0.91, "loss": 0.3}, path=log)
    log_experiment("c", {"loss": 0.2}, path=log)
    assert best_experiment("accuracy", "max", log)["name"] == "b"
    assert best_experiment("loss", "min", log)["name"] == "c"
    assert best_experiment("missing", "max", log) is None
    assert len(read_experiments(log)) == 3
    code, out = run_cli("experiments", "--path", log, "--best", "accuracy")
    assert code == 0 and json.loads(out)["name"] == "b"


def test_invalid_experiment_mode_rejected(tmp_path):
    try:
        best_experiment("loss", "median", str(tmp_path / "x.jsonl"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_monitor_command_on_empty_log(tmp_path):
    code, out = run_cli("monitor", "--log", str(tmp_path / "none.jsonl"))
    assert code == 0 and json.loads(out)["sample_count"] == 0


def test_chatbot_new_topics():
    from modern.chatbot import answer
    assert "leak" in answer("What is data leakage?").lower()
    assert "pretrained" in answer("What is transfer learning?").lower()
