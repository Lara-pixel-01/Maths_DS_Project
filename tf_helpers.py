from pathlib import Path

_src = Path(__file__).with_name("text_features.py").read_text(encoding="utf-8")
H = {"__name__": "text_features_helpers", "__file__": "text_features.py"}
exec(compile(_src.split('pd.set_option("display.width", 220)')[0], "text_features.py", "exec"), H)