# video_logs_lane 目录说明

这个目录现在采用和 `training_runs` 类似的分类方式来保存测试视频，并且会预先创建固定目录骨架。

## 固定目录结构

```text
video_logs_lane/
  base/
    highway_standard/
    highway_dense/
    merge/
    roundabout/
  ours/
    highway_standard/
    highway_dense/
    merge/
    roundabout/
```

含义如下：

- `base/`：基础算法（baseline）的视频
- `ours/`：改进算法（ours）的视频
- 四种场景目录：
  - `highway_standard/`
  - `highway_dense/`
  - `merge/`
  - `roundabout/`

## 为什么现在 `base` 和 `ours` 下会有四个子目录

现在代码会自动预创建这 8 个目录，不需要等你先录某个场景才出现。

这样做是为了让视频目录和 `training_runs/` 保持一致，方便你一眼区分：

- 算法类别：`base` / `ours`
- 场景类别：四种场景分别存放

## 为什么有的场景目录看起来是空的

如果某个场景还没有录过视频，那么该目录里可能暂时只有一个占位文件：

```text
.gitkeep
```

这表示目录结构已经准备好了，只是这个场景还没有生成录像。

## 实际录像怎么保存

每次录制时，会在对应场景目录下继续创建时间戳子目录，例如：

```text
video_logs_lane/ours/merge/20260331_170529_070225/
```

视频文件名会带上算法类别、场景、测试标签和保存时间，例如：

```text
ours_merge_test_failure020-seed1_20260331_170529_070225-episode-0.mp4
base_highway_dense_test_demo_20260331_171000_123456-episode-0.mp4
```

## 当前保存规则

1. 先区分算法：`base/` 或 `ours/`
2. 再区分场景：四种场景分别存放
3. 再区分录制批次：按时间戳创建独立目录
4. 最后保存多个视频文件：不会覆盖历史录像
