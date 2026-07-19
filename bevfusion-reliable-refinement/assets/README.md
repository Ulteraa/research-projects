# Media assets

Large videos are intentionally excluded from Git. A compressed homepage demo can be added here as:

```text
objectrefine_ep4_rot_web.mp4
```

Recommended web encoding:

```bash
ffmpeg -i objectrefine_ep4_rot.mp4 \
  -vf "scale=1280:-2,fps=15" \
  -c:v libx264 -preset slow -crf 30 -pix_fmt yuv420p \
  -movflags +faststart -an \
  objectrefine_ep4_rot_web.mp4
```
