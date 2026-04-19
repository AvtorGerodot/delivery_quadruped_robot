# delivery_b2 — Unitree B2 в Genesis

Проект содержит окружение, RL-политики и высокоуровневый Python-API для
симулированного четвероногого робота **Unitree B2** в движке
[Genesis](./Genesis). Все зависимости управляются через `uv`.

В проекте есть две независимые тренируемые политики:


| Политика   | Файл среды                   | Файл обучения                  | Логика                                                                                                   |
| ---------- | ---------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `ball`     | `src/examples/b2_env.py`     | `src/examples/b2_train.py`     | Робот гонится за виртуальным «красным шариком». Команда — желаемая мировая точка `(x, y)` и yaw корпуса. |
| `velocity` | `src/examples/b2_vel_env.py` | `src/examples/b2_train_vel.py` | Робот отслеживает скоростную команду `(lin_vel_x, lin_vel_y, ang_vel_yaw)` — классическая go2-локомоция. |


Для обеих политик есть одинаково выглядящий интерфейс инференса —
`src/api.py::Robot` (см. ниже и [api.md](./api.md)).

---

## 1. Установка

### Предварительные требования

- Linux x86_64 (тестировалось на Ubuntu 22.04 / 24.04).
- Python `>=3.11,<3.14` (Genesis требует `<3.14`).
- `uv` — менеджер зависимостей. Ставится в одну команду:
`curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **Для GPU-обучения:** NVIDIA GPU + драйвер, совместимый с CUDA 12.4
(проверить `nvidia-smi`).

### Клонирование подмодулей

В `pyproject.toml` прописаны локальные зависимости на соседние
репозитории:

```
/
├── Genesis/                     симуляционный движок (editable install)
├── unitree_ros/                 URDF-описания роботов (b2_description, z1_description)
├── unitree_sdk2_python/         Python-SDK для реального робота (editable install)
└── cyclonedds/                  транспорт для SDK (опционально)
```

Если подмодули ещё не получены:

```bash
git submodule update --init --recursive
```

### Синхронизация зависимостей

```bash
uv sync
```

`uv` подтянет:

- `torch==2.6.0` и `torchvision==0.21.0` с индекса `pytorch-cu124`
(колёсики CUDA 12.4, работают с драйверами `>= 550`).
- `genesis-world` (editable) из `./Genesis`.
- `rsl-rl-lib==2.2.4` — строго эту версию; новее (`>=5`) имеют
несовместимый API.
- `tensorboard`, `pygame>=2.5` (для DualShock 4), `unitree-sdk2py`.

Если GPU отсутствует и нужен CPU-билд Torch — замените индекс в
`pyproject.toml` с `pytorch-cu124` на `pytorch-cpu`
(`https://download.pytorch.org/whl/cpu`) и повторите `uv sync`.

### Быстрая проверка

```bash
uv run python -c "import genesis as gs; gs.init(backend=gs.cpu); print('Genesis OK')"
uv run python src/examples/b2_train.py --help
uv run python src/examples/b2_train_vel.py --help
```

---

## 2. Обучение политик

Все тренировки пишут логи в `logs/<exp_name>/` — веса модели
`model_<it>.pt`, дамп конфигов `cfgs.pkl`, TensorBoard events. Удалите
эту папку, чтобы начать с нуля.

### 2.1. Ball-политика (`b2_train.py`)

Адаптивное расписание из `dynamic_ics_dog_train.py`: сначала робот
учится стоять, когда `stability` award выходит на плато, коэффициент
`target_coeff` плавно поднимается, и робот начинает преследовать
виртуальный шарик.

```bash
uv run src/examples/b2_train.py \
    --exp_name b2-target-rl \
    --num_envs 4096 \
    --max_iterations 500 \
    --backend gpu \
    --device cuda
```

Флаг `--device` — торч-устройство для PPO (например `cuda:0`). При
`--backend cpu` автоматически ставится `cpu`.

### 2.2. Velocity-политика (`b2_train_vel.py`)

Простой PPO-цикл в стиле `go2_train.py`, без адаптивного расписания.
Диапазоны команд берутся из `default_cfgs()` (по умолчанию — точные go2-значения,
`lin_vel_x ∈ [0.5, 0.5]` и пр., то есть робот научится ходить только
вперёд). Чтобы получить всенаправленную политику, расширьте диапазоны с
CLI:

```bash
uv run src/examples/b2_train_vel.py \
    --exp_name b2-walk-omni \
    --lin_vel_x_range -1.0 1.0 \
    --lin_vel_y_range -0.5 0.5 \
    --ang_vel_range -1.0 1.0 \
    --num_envs 4096 \
    --max_iterations 800 \
    --backend gpu --device cuda
```

После обучения эти же диапазоны будут зашиты в `cfgs.pkl` и
автоматически применятся при инференсе (см. ниже).

### 2.3. Мониторинг обучения

```bash
uv run tensorboard --logdir logs/
```

Ключевые скаляры:

- `Train/mean_reward` — общая награда.
- `Episode/rew_*` — декомпозиция по компонентам.
- Только для ball-политики: `Dynamic/target_coeff`, `Dynamic/phase`
(0 = stability, 1 = targeting), `Dynamic/plateau_count`.

---

## 3. Инференс

### 3.1. Клавиатурный viewer — `b2_eval.py`

Сам Genesis-вьювер + стрелочное управление.

```bash
# Ball-политика:
uv run src/examples/b2_eval.py --mode ball     -e b2-target-rl --ckpt 499

# Velocity-политика:
uv run src/examples/b2_eval.py --mode velocity -e b2-walk-omni --ckpt 799
```

`--ckpt -1` (по умолчанию) — взять самую свежую модель из
`logs/<exp>/`. Раскладка клавиш печатается в терминал при старте.

### 3.2. DualShock 4 — `ds4_control.py`

```bash
# Ball:
uv run src/examples/ds4_control.py --mode ball     -e b2-target-rl
# Velocity:
uv run src/examples/ds4_control.py --mode velocity -e b2-walk-omni --ckpt 799
```

Левый стик → поступательное движение (XY), правый X → yaw, `Circle` → стоп,
`Options` → выход. В `velocity`-моде значения стика маппятся **в точные
тренировочные диапазоны**, прочитанные из `cfgs.pkl`.

### 3.3. Программный API — `src/api.py`

Для скриптов, демо, интеграций. Интерфейс одинаков для обеих политик:

```python
from api import Robot

# Ball policy:
r = Robot(exp_name="b2-target-rl", mode="ball", show_viewer=True)
r.move(1.0, 0.0, time=3.0)          # дойти к (1, 0) за ~3 с
r.rotate(math.pi / 2, time=2.0)     # повернуться на +90° за ~2 с
r.close()

# Velocity policy (ровно тот же код):
r = Robot(exp_name="b2-walk-omni", mode="velocity", show_viewer=True)
r.move(1.0, 0.0, time=3.0)
r.rotate(math.pi / 2, time=2.0)
r.close()
```

### 3.4 Ленивый api

Требует предварительно обученной "скоростной" политики

```bash
uv run uv run src/api.py
```

Подробнее — в [api.md](./api.md).

---

## 4. Расширение: добавление манипулятора Unitree Z1

В репозитории уже есть `unitree_ros/robots/z1_description`. План
расширения: собрать через xacro единый URDF `B2 + Z1` в папке
`complex_urdf/`, а политику RL **не переобучать** — она и так отвечает
только за 12 суставов ног. Управление рукой идёт вручную из Python
(прямой запись в DOF манипулятора), как и планируется.

Что конкретно нужно поменять в коде:

### 4.1. Собрать единый URDF

Создайте `complex_urdf/b2_with_z1.xacro`, в который подключите B2 и Z1
как `xacro:include` и прикрепите базу Z1 к нужному линку B2 (например,
`trunk`) через `<joint type="fixed">`. Соберите URDF командой:

```bash
xacro complex_urdf/b2_with_z1.xacro > complex_urdf/b2_with_z1.urdf
```

Файл должен валидироваться `check_urdf` без ошибок.

### 4.2. Подменить путь URDF

В `src/examples/b2_env.py` и `src/examples/b2_vel_env.py` в начале файла:

```python
B2_URDF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "complex_urdf", "b2_with_z1.urdf")
)
```

Больше в этих файлах ничего править не нужно: `env_cfg["joint_names"]`
по-прежнему перечисляет **только 12 суставов ног**, поэтому
`motors_dof_idx`, `control_dofs_position(..., slice(6, 18))` и
наблюдения политики остаются теми же. Обсервации манипулятора в
policy-обзор не попадают — сеть, обученная на чисто-ноговой модели,
продолжит работать без переобучения.

### 4.3. Инициализировать манипулятор в ресете

В `_apply_reset(...)` (оба env'а) добавьте блок после установки
позиций ног:

```python
ARM_JOINT_NAMES = [
    "Joint01", "Joint02", "Joint03", "Joint04", "Joint05", "Joint06",
    # плюс "jointGripper", если используется
]
ARM_DEFAULT_DOF_POS = torch.tensor(
    [0.0, 1.5, -1.0, 0.0, 0.0, 0.0],
    dtype=gs.tc_float, device=self.device,
)
# one-time в __init__:
self.arm_dof_idx = torch.tensor(
    [self.robot.get_joint(n).dof_start for n in ARM_JOINT_NAMES],
    dtype=gs.tc_int, device=self.device,
)
self.robot.set_dofs_kp([200.0]*len(ARM_JOINT_NAMES), self.arm_dof_idx)
self.robot.set_dofs_kv([8.0]*len(ARM_JOINT_NAMES),   self.arm_dof_idx)

# в _apply_reset:
self.robot.set_dofs_position(
    position=ARM_DEFAULT_DOF_POS.unsqueeze(0).expand(n, -1).contiguous(),
    dofs_idx_local=self.arm_dof_idx,
    zero_velocity=True,
    envs_idx=envs_idx,
)
```

### 4.4. Управлять рукой вручную

Политика RL в этот массив DOF не пишет — в `step()` она вызывает
`self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))`,
что затрагивает **только** суставы ног. Для руки в обычном цикле
симуляции каждый тик задавайте желаемые углы:

```python
# пример: выставить руку в «шагающее» безопасное положение
arm_target = torch.tensor([[0.0, 1.5, -1.0, 0.0, 0.0, 0.0]],
                          dtype=gs.tc_float, device=env.device)
env.robot.control_dofs_position(arm_target, env.arm_dof_idx)
```

Это можно спрятать в новом методе `Robot.set_arm_joints(values)` в
`src/api.py` — он будет дёргать `self.env.robot.control_dofs_position`
перед каждым `step()`. Шаблон (добавить в класс `Robot`):

```python
def set_arm_joints(self, values: "Sequence[float]") -> None:
    """Записать желаемые углы 6 суставов Z1. Применяется на каждом шаге."""
    self._arm_target = torch.as_tensor(
        values, dtype=gs.tc_float, device=self.env.device
    ).unsqueeze(0)

# в _BallBackend.step / _VelocityBackend.step — перед actions = self._policy(...):
if getattr(self, "_arm_target", None) is not None:
    self.env.robot.control_dofs_position(self._arm_target, self.env.arm_dof_idx)
```

### 4.5. Если нужна IK/FK для конечной точки

Для обратной кинематики Z1 есть `unitree_ros/robots/z1_description/meshes`

- стандартные библиотеки: `pinocchio`, `kinpy`, `roboticstoolbox-python`
или IK внутри Genesis (`robot.get_link("<ee>").inverse_kinematics(...)`).
Решение IK даёт `target_joint_angles`, которые передаются в
`set_arm_joints(...)`. RL-политика про руку всё ещё ничего не знает.

### 4.6. Наконечная камера

Когда на ef-линке Z1 появляется камера, Genesis поддерживает её через
`scene.add_camera(link=robot.get_link("camera_link"), ...)`. Снимки
берутся `camera.render()` — RL-политика остаётся без изменений.

---

## 5. Устранение типичных проблем


| Симптом                                                                                  | Причина / решение                                                                                                                                                                                            |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `RuntimeError: Reference at 'refs/heads/master' does not exist` при старте `b2_train.py` | В `b2_train.py` это ловится try/except (снимок git-state). Если «утекло» — сделайте `git commit --allow-empty -m "bootstrap"` в корне репозитория.                                                           |
| `[Genesis] [WARNING] Neutral robot position (qpos0) exceeds joint limits`                | Косметика, идёт от URDF B2. На обучение и симуляцию не влияет.                                                                                                                                               |
| `RuntimeError: normal expects all elements of std >= 0.0` при обучении                   | Это NaN в политике. Проверьте, что env прошёл smoke-test: `uv run python -c "from b2_env import B2TargetEnv, default_cfgs; ..."`. Фикс обычно в правильном сбросе DOF (см. коммит, вводящий `_apply_reset`). |
| `No gamepad detected` в `ds4_control.py`                                                 | Проверьте, что DS4 виден в `/dev/input`. На Linux ≥5.12 должен подхватиться модулем `hid-playstation`. Если нет — `sudo modprobe hid_playstation` или `ds4drv`.                                              |
| `--device=cuda requested but torch.cuda.is_available() is False`                         | Установлены CPU-колёсики Torch. Замените в `pyproject.toml` индекс `pytorch-cpu` на `pytorch-cu124` и повторите `uv sync`.                                                                                   |


---

## 6. Структура проекта

```
delivery_b2/
├── Genesis/                                 симулятор (editable)
├── unitree_ros/robots/{b2,z1}_description/  URDF-модели
├── unitree_sdk2_python/                     SDK реального робота
├── complex_urdf/                            (пока пусто) B2+Z1 xacro/urdf
├── src/
│   ├── api.py                               высокоуровневый Robot
│   ├── example_b2_teleop.py                 классическая телеоперация без RL
│   └── examples/
│       ├── b2_env.py            среда ball-политики (виртуальный шарик)
│       ├── b2_train.py          тренировка ball-политики
│       ├── b2_eval.py           клавиатурный eval (оба мода)
│       ├── b2_vel_env.py        среда velocity-политики (скоростная команда)
│       ├── b2_train_vel.py      тренировка velocity-политики
│       └── ds4_control.py       управление с DualShock 4 (оба мода)
├── pyproject.toml
└── README.md (этот файл)
```

Дополнительное руководство по работе с `api.py` — в [api.md](./api.md).