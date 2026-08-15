import sys
path=sys.argv[1]
content=open(sys.argv[2],encoding="utf-8").read()
open(path,"w",encoding="utf-8").write(content)
print(f"Written {len(content)} chars to {path}")
