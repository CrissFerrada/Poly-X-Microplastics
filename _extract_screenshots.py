import re, base64, os
os.makedirs('manual_screenshots', exist_ok=True)
with open('Manual_PolyX.html', 'r', encoding='utf-8') as f:
    html = f.read()
pattern = re.compile(r"""<img[^>]*src=['"]data:image/(\w+);base64,([^'"]+)['"][^>]*>""", re.IGNORECASE)
matches = list(pattern.finditer(html))
print(f'Total imagenes embebidas: {len(matches)}')
for i, m in enumerate(matches, 1):
    ext = m.group(1).lower()
    data = base64.b64decode(m.group(2))
    after = html[m.end():m.end() + 500]
    cap_m = re.search(r"""class=['"]caption['"][^>]*>(.*?)<""", after, re.DOTALL)
    cap = ''
    if cap_m:
        cap = re.sub(r'\s+', ' ', cap_m.group(1)).strip()[:60]
    safe_cap = re.sub(r'[^\w\-]+', '_', cap).strip('_')
    name = f'fig_{i:02d}_{safe_cap}.{ext}' if safe_cap else f'fig_{i:02d}.{ext}'
    with open(os.path.join('manual_screenshots', name), 'wb') as out:
        out.write(data)
    safe_print = cap.encode('ascii', 'replace').decode('ascii')
    print(f'  {i:02d} -> {name} ({len(data)//1024} KB)  caption="{safe_print}"')
