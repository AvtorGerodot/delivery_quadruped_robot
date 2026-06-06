# Гибридные роботы (собака + манипулятор)

Шпаргалка по командам трейна/эвала. Есть две «собаки с рукой»:

| Робот | URDF | Кто это |
|-------|------|---------|
| **b2_z1** | `complex_urdf/b2_z1.urdf` | Unitree B2 + Z1, собранный нами (без гриппера) |
| **delivery_dog** | `complex_urdf/delivery_dog_b2_z1.urdf` | «улучшенная» модель заказчика из `delivery_dog_ws` (B2 + Z1 + гриппер + коробка) |

Для каждой собаки два режима:
- **velocity (фикс. манипулятор)** — рука заморожена, политика учит только ходьбу по целевой скорости.
- **WBC (whole-body control)** — ноги + рука вместе тянут концевую точку к 3D-шарику.

Перед обучением окружение можно собрать через `uv` (`uv sync`), все команды — через `uv run`.

---

## 1. b2_z1 (наша сборка B2 + Z1)

Один раз собрать URDF:
```bash
uv run complex_urdf/build_b2_z1.py
```

### velocity (фиксированный манипулятор)
```bash
# трейн
uv run src/examples/b2_train_vel.py -e b2z1-walk --robot b2_z1 \
    --backend gpu --device cuda \
    --lin_vel_x_range -1.0 1.0 --lin_vel_y_range -0.5 0.5 --ang_vel_range -1.0 1.0 \
    -B 4096 --max_iterations 3000

# эвал (управление стрелочками)
uv run src/examples/b2_eval.py --mode velocity -e b2z1-walk
```

### WBC (рука тянется к шарику)
```bash
# трейн
uv run src/examples/b2_z1_wbc_train.py -e b2z1-wbc \
    --backend gpu --device cuda -B 4096 --max_iterations 2000

# эвал (управление 3D-шариком)
uv run src/examples/b2_z1_wbc_eval.py -e b2z1-wbc --ckpt -1
```

---

## 2. delivery_dog (модель заказчика, эвал в сцене подъезда)

Один раз почистить/собрать URDF и (по желанию) проверить спавн:
```bash
uv run complex_urdf/build_delivery_dog.py
uv run src/examples/spawn_delivery_dog.py --entrance --entrance-x 2.2   # визуальная проверка
```

### velocity (фиксированный манипулятор + гриппер)
```bash
# трейн (подъезд НЕ участвует)
uv run src/examples/delivery_dog_train_vel.py -e dd-walk \
    --backend gpu --device cuda \
    --lin_vel_x_range -1.0 1.0 --lin_vel_y_range -0.5 0.5 --ang_vel_range -1.0 1.0 \
    -B 4096 --max_iterations 3000

# эвал в сцене подъезда (стрелочки/Q-E), вход сдвинут на 2.2 м вперёд
uv run src/examples/delivery_dog_eval_vel.py -e dd-walk --ckpt -1 --entrance-x 2.2
```

### WBC (рука тянется к шарику, гриппер статичен)
```bash
# трейн (подъезд НЕ участвует)
uv run src/examples/delivery_dog_wbc_train.py -e dd-wbc \
    --backend gpu --device cuda -B 4096 --max_iterations 2000

# эвал в сцене подъезда (3D-шарик)
uv run src/examples/delivery_dog_wbc_eval.py -e dd-wbc --ckpt -1 --entrance-x 2.2
```

---

### Подсказки
- `--ckpt -1` — взять последний сохранённый чекпойнт; либо число (например `--ckpt 1200`).
- `--entrance-x` — на сколько метров отодвинуть подъезд вперёд, чтобы робот не спавнился в дверях. Значение по умолчанию — в `src/examples/delivery_dog_cfgs.py` (`ENTRANCE_OFFSET_X`).
- `--no-entrance` в эвал-скриптах delivery_dog — эвал на голой плоскости без подъезда.
- Логи и чекпойнты пишутся в `logs/<exp_name>/`.
