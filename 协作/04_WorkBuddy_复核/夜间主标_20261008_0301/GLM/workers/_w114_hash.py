# -*- coding: utf-8 -*-
"""分片114 哈希自检脚本：精确使用清单中的路径与哈希，不做任何修改。"""
import hashlib

# 清单权威值：item_id -> (image_path, image_sha256)
MANIFEST = [
    ("wbg0792",
     "D:/pycode/晶圆图研究/data/images/wafer_00042242_007.png",
     "d952e32cf646eec3e7edf30a0cf0506a8ef33344568b295f4d1e8eef05849472"),
    ("wbg0793",
     "D:/pycode/晶圆图研究/data/images/wafer_00042253_006.png",
     "f28a8909747073aa71f5a50b0fc09ee09da1dc591bc8bb7926aebc5f259c25e7"),
    ("wbg0794",
     "D:/pycode/晶圆图研究/data/images/wafer_00042259_024.png",
     "05e7d719a7b5f482a3da615e596c04ea32af406ca684ff0e1df15c0f7c82b2ce"),
    ("wbg0795",
     "D:/pycode/晶圆图研究/data/images/wafer_00042265_015.png",
     "6cf4db9bb5fba7cb8e94632ac679b540d778e002faa0a25985f95d5f7581bdae"),
    ("wbg0796",
     "D:/pycode/晶圆图研究/data/images/wafer_00042269_014.png",
     "29000cdc1e1542457b1409af2576e27839ad0f39c6c38a6c58ee579ccdfbb1d1"),
    ("wbg0797",
     "D:/pycode/晶圆图研究/data/images/wafer_00042271_016.png",
     "dde5f552ace37b52f8300e3ff8aa6da640909ef5ffc8b765c94603866f09d520"),
    ("wbg0798",
     "D:/pycode/晶圆图研究/data/images/wafer_00042277_013.png",
     "8e5797b07cc5fc1ddac356f7f836bb073ac61a700d4406a2da750df32df7eafa"),
]

def main():
    print("=" * 72)
    for item_id, path, expected in MANIFEST:
        assert len(expected) == 64, f"{item_id} manifest sha len != 64"
        try:
            with open(path, "rb") as f:
                actual = hashlib.sha256(f.read()).hexdigest()
        except OSError as exc:
            print(f"{item_id}  READ_FAIL  {exc}")
            continue
        status = "MATCH" if actual == expected else "MISMATCH"
        print(f"{item_id}  {status}")
        if status == "MISMATCH":
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")
    print("=" * 72)

if __name__ == "__main__":
    main()
