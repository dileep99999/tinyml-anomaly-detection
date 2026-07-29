import json
import os

notebook_path = r"c:\Users\lrlacm\Music\tinyML-anomaly-detection\notebooks\09_arduino_uno_compatibility.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

new_source = [
    "# Estimate memory footprint on target MCU boards and save compatibility plot\n",
    "mcus = [\n",
    "    {'name': 'Arduino Nano 33 BLE',  'flash_kb': 1024,  'ram_kb': 256,  'cpu': 'Cortex-M4 64MHz'},\n",
    "    {'name': 'STM32F411',            'flash_kb': 512,   'ram_kb': 128,  'cpu': 'Cortex-M4 100MHz'},\n",
    "    {'name': 'STM32F446',            'flash_kb': 512,   'ram_kb': 128,  'cpu': 'Cortex-M4 180MHz'},\n",
    "    {'name': 'STM32H743',            'flash_kb': 2048,  'ram_kb': 1024, 'cpu': 'Cortex-M7 480MHz'},\n",
    "    {'name': 'ESP32',                'flash_kb': 4096,  'ram_kb': 520,  'cpu': 'Xtensa LX6 240MHz'},\n",
    "    {'name': 'Arduino Uno (ATmega)', 'flash_kb': 32,    'ram_kb': 2,    'cpu': 'AVR 16MHz'},\n",
    "]\n",
    "\n",
    "model_flash_kb = len(tflite_f32) / 1024\n",
    "runtime_flash_kb = 12.0    # EloquentTinyML/AVR runtime \u2248 12 KB\n",
    "total_flash_kb = model_flash_kb + runtime_flash_kb\n",
    "total_ram_kb   = 1.0 + 0.5   # 1.0 KB Arena + 0.5 KB Stack\n",
    "\n",
    "print(f'Tiny Model requires:')\n",
    "print(f'  Flash : {total_flash_kb:.2f} KB  (model {model_flash_kb:.2f} + runtime {runtime_flash_kb:.2f} KB)')\n",
    "print(f'  RAM   : {total_ram_kb:.2f} KB  (arena 1.0 + stack 0.5 KB)')\n",
    "print()\n",
    "print(f'{\"MCU\":<25} {\"Flash\":>8} {\"RAM\":>8} {\"Flash OK?\":>10} {\"RAM OK?\":>10}')\n",
    "print(\"-\" * 65)\n",
    "for m in mcus:\n",
    "    flash_ok = '\\u2705 YES' if m['flash_kb'] >= total_flash_kb else '\\u274c NO'\n",
    "    ram_ok   = '\\u2705 YES' if m['ram_kb']   >= total_ram_kb   else '\\u274c NO'\n",
    "    print(f\"{m['name']:<25} {m['flash_kb']:>7}K {m['ram_kb']:>7}K {flash_ok:>10} {ram_ok:>10}\")\n",
    "\n",
    "# Visual MCU compatibility chart\n",
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "names    = [m['name'] for m in mcus]\n",
    "flash_kb = [m['flash_kb'] for m in mcus]\n",
    "ram_kb   = [m['ram_kb']   for m in mcus]\n",
    "x  = np.arange(len(names))\n",
    "w  = 0.35\n",
    "\n",
    "b1 = ax.bar(x - w/2, flash_kb, w, label='Flash (KB)', color='#2196F3', alpha=0.8)\n",
    "b2 = ax.bar(x + w/2, ram_kb,   w, label='RAM (KB)',   color='#4CAF50', alpha=0.8)\n",
    "\n",
    "ax.axhline(total_flash_kb, color='#2196F3', linestyle='--', linewidth=2,\n",
    "            label=f'Required Flash ({total_flash_kb:.2f} KB)')\n",
    "ax.axhline(total_ram_kb, color='#4CAF50', linestyle='--', linewidth=2,\n",
    "            label=f'Required RAM ({total_ram_kb:.2f} KB)')\n",
    "\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)\n",
    "ax.set_ylabel('Memory (KB)')\n",
    "ax.set_title('MCU Memory vs Tiny Model Requirements (Arduino Uno Compatible)')\n",
    "ax.legend(fontsize=8)\n",
    "ax.set_yscale('log')\n",
    "ax.grid(True, axis='y', alpha=0.3)\n",
    "plt.tight_layout()\n",
    "\n",
    "plots_dir = '../results/plots'\n",
    "os.makedirs(plots_dir, exist_ok=True)\n",
    "plt.savefig(os.path.join(plots_dir, 'mcu_compatibility_uno.png'), dpi=150)\n",
    "plt.show()\n",
    "\n",
    "# Export to C byte array for the Arduino IDE\n",
    "c_header = []\n",
    "c_header.append('// Arduino Uno compatible tiny anomaly detection model')\n",
    "c_header.append(f'// Size: {len(tflite_f32)} bytes')\n",
    "c_header.append('#include <stdint.h>')\n",
    "c_header.append('const unsigned char uno_anomaly_model[] = {')\n",
    "\n",
    "hex_vals = [f'0x{b:02x}' for b in tflite_f32]\n",
    "for i in range(0, len(hex_vals), 12):\n",
    "    chunk = ', '.join(hex_vals[i:i+12])\n",
    "    c_header.append(f'  {chunk},')\n",
    "c_header.append('};')\n",
    "\n",
    "with open('../models/uno_anomaly_model.h', 'w') as f:\n",
    "    f.write('\\n'.join(c_header))\n",
    "\n",
    "print('Saved model C header file to: models/uno_anomaly_model.h')\n",
    "print('\\nFirst 10 lines of generated C header:')\n",
    "print('\\n'.join(c_header[:10]))"
]

cell_updated = False
for cell in notebook.get("cells", []):
    source_text = "".join(cell.get("source", []))
    if "# Export to C byte array for the Arduino IDE" in source_text and "mcus" not in source_text:
        cell["source"] = new_source
        cell_updated = True
        break

if cell_updated:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print("Successfully updated cell in notebook.")
else:
    print("Cell not found or already updated.")
