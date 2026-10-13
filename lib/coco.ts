/**
 * COCO metadata shared by both models.
 * - Segmentation (yolo26n-seg) predicts the 80 COCO object classes.
 * - Pose (yolo26n-pose) predicts the single "person" class + 17 keypoints.
 */

export const COCO_CLASSES: string[] = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
  "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
  "toothbrush",
];

export const KEYPOINT_NAMES: string[] = [
  "nose", "left eye", "right eye", "left ear", "right ear",
  "left shoulder", "right shoulder", "left elbow", "right elbow",
  "left wrist", "right wrist", "left hip", "right hip",
  "left knee", "right knee", "left ankle", "right ankle",
];

/** Bone connections, 0-indexed (COCO 17-keypoint layout). */
export const SKELETON: [number, number][] = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
  [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [9, 11],
  [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
];

/** A distinct, readable palette cycled across classes. */
export const PALETTE: string[] = [
  "#ef4444", "#f97316", "#f59e0b", "#eab308", "#84cc16", "#22c55e", "#10b981",
  "#14b8a6", "#06b6d4", "#0ea5e9", "#3b82f6", "#6366f1", "#8b5cf6", "#a855f7",
  "#d946ef", "#ec4899", "#f43f5e", "#78716c", "#0d9488", "#4d7c0f",
];

export function classColor(classId: number): string {
  return PALETTE[classId % PALETTE.length];
}

export const COCO_KEYPOINT_COUNT = 17;
