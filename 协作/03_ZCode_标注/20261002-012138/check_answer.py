import json, hashlib, sys, time, os
sample_id = sys.argv[1]
d = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(d, sample_id + "_raw.json")
b = open(raw_path, "rb").read()
res = {"sample_id": sample_id, "raw_bytes": len(b), "raw_sha256": hashlib.sha256(b).hexdigest()}
try:
    obj = json.loads(b.decode("utf-8")); res["json_parse_ok"] = True
except Exception as e:
    res["json_parse_ok"] = False; res["parse_error"] = repr(e); res["format_pass"] = False
    json.dump(res, open(os.path.join(d, sample_id + "_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("FAIL json_parse"); sys.exit(0)
expected_keys = ["defect_class","morphology","radial_zone","clock_direction","extent_r","caption_zh","uncertainty"]
class_enum = {"Center","Donut","Edge_Loc","Edge_Ring","Loc","Near_full","Random","Scratch","none","unknown"}
zone_enum = {"center","middle","edge","global","unknown","none"}
checks = {
  "keys_exact_order": list(obj.keys()) == expected_keys,
  "defect_class_enum": obj.get("defect_class") in class_enum,
  "morphology_str_nonempty": isinstance(obj.get("morphology"), str) and obj.get("morphology","").strip() != "",
  "radial_zone_enum": obj.get("radial_zone") in zone_enum,
  "clock_direction_null_or_str": obj.get("clock_direction") is None or isinstance(obj.get("clock_direction"), str),
  "extent_r_is_null": obj.get("extent_r") is None,
  "caption_zh_str_nonempty": isinstance(obj.get("caption_zh"), str) and obj.get("caption_zh","").strip() != "",
  "uncertainty_str": isinstance(obj.get("uncertainty"), str),
}
res["n_fields"] = len(obj)
res["field_checks"] = checks
res["format_pass"] = all(checks.values())
res["checked_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
json.dump(res, open(os.path.join(d, sample_id + "_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("format_pass:", res["format_pass"])
for k, v in checks.items(): print(" ", k, "=", v)
