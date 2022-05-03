# MS COCO Dataset
import numpy as np
from matplotlib import pyplot as plt

import coco_train as coco
from neural_network.models.mrcnn import visualise, utils
import neural_network.models.mrcnn.model as modellib

config = coco.CocoConfig()

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class Prediction:
    config = InferenceConfig()
    config.display()

    # Directory to save logs and trained model
    MODEL_DIR = "neural_network/logs"

    dataset_classes = ['BG', 'person', 'bicycle', 'car', 'motorcycle',
                       'airplane', 'bus', 'train', 'truck', 'boat',
                       'traffic light', 'fire hydrant', 'stop sign',
                       'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra',
                       'giraffe', 'backpack', 'umbrella', 'handbag',
                       'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard',
                       'sports ball', 'kite', 'baseball bat',
                       'baseball glove', 'skateboard', 'surfboard',
                       'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana',
                       'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot',
                       'hot dog', 'pizza', 'donut', 'cake', 'chair',
                       'couch', 'potted plant', 'bed', 'dining table',
                       'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven',
                       'toaster', 'sink', 'refrigerator', 'book',
                       'clock',
                       'vase', 'scissors', 'teddy bear', 'hair drier',
                       'toothbrush']

    def get_ax(self, rows=1, cols=1, size=16):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.

        Adjust the size attribute to control how big to render images
        """
        _, ax = plt.subplots(rows, cols,
                             figsize=(size * cols, size * rows))
        return ax

    def precision_recall(self, model, image):
        # Run object detection
        results = model.detect([image], verbose=1)

        # Display results
        ax = self.get_ax(1)
        r = results[0]
        visualise.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    self.dataset_classes, r['scores'], ax=ax,
                                    title="Predictions")

        # Draw precision-recall curve
        AP, precisions, recalls, overlaps = utils.compute_ap(r['rois'], r['class_ids'], r['masks'],
                                                             r['rois'], r['class_ids'], r['scores'], r['masks'])
        visualise.plot_precision_recall(AP, precisions, recalls)

        # Grid of ground truth objects and their predictions
        visualise.plot_overlaps(r['class_ids'], r['class_ids'], r['scores'],
                        overlaps, self.dataset_classes)


    def proposal_classification(self, model, image):
        # Get input and output to classifier and mask heads.
        mrcnn = model.run_graph([image], [
            ("proposals", model.keras_model.get_layer("ROI").output),
            (
                "probs",
                model.keras_model.get_layer("mrcnn_class").output),
            (
                "deltas",
                model.keras_model.get_layer("mrcnn_bbox").output),
            ("masks", model.keras_model.get_layer("mrcnn_mask").output),
            ("detections",
             model.keras_model.get_layer("mrcnn_detection").output),
        ])

        # Get detection class IDs. Trim zero padding.
        det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
        print(det_class_ids)
        det_count = np.where(det_class_ids == 0)[0][0]
        det_class_ids = det_class_ids[:det_count]
        detections = mrcnn['detections'][0, :det_count]

        print("{} detections: {}".format(
            det_count, np.array(self.dataset_classes)[det_class_ids]))

        captions = ["{} {:.3f}".format(self.dataset_classes[int(c)],
                                       s) if c > 0 else ""
                    for c, s in zip(detections[:, 4], detections[:, 5])]
        visualise.draw_boxes(
            image,
            refined_boxes=utils.denorm_boxes(detections[:, :4],
                                             image.shape[:2]),
            visibilities=[2] * len(detections),
            captions=captions, title="Detections",
            ax=self.get_ax())

        # Proposals are in normalized coordinates. Scale them
        # to image coordinates.
        h, w = self.config.IMAGE_SHAPE[:2]
        proposals = np.around(
            mrcnn["proposals"][0] * np.array([h, w, h, w])).astype(
            np.int32)

        # Class ID, score, and mask per proposal
        roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
        roi_scores = mrcnn["probs"][
            0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
        roi_class_names = np.array(self.dataset_classes)[
            roi_class_ids]
        roi_positive_ixs = np.where(roi_class_ids > 0)[0]

        # How many ROIs vs empty rows?
        print("{} Valid proposals out of {}".format(
            np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
        print("{} Positive ROIs".format(len(roi_positive_ixs)))

        # Class counts
        print(
            list(zip(*np.unique(roi_class_names, return_counts=True))))

        # Display a random sample of proposals.
        # Proposals classified as background are dotted, and
        # the rest show their class and confidence score.
        limit = 200
        ixs = np.random.randint(0, proposals.shape[0], limit)
        captions = ["{} {:.3f}".format(self.dataset_classes[c],
                                       s) if c > 0 else ""
                    for c, s in
                    zip(roi_class_ids[ixs], roi_scores[ixs])]
        visualise.draw_boxes(image, boxes=proposals[ixs],
                             visibilities=np.where(
                                 roi_class_ids[ixs] > 0, 2, 1),
                             captions=captions,
                             title="ROIs Before Refinement",
                             ax=self.get_ax())

        # Class-specific bounding box shifts.
        roi_bbox_specific = mrcnn["deltas"][
            0, np.arange(proposals.shape[0]), roi_class_ids]
        modellib.log("roi_bbox_specific", roi_bbox_specific)

        # Apply bounding box transformations
        # Shape: [N, (y1, x1, y2, x2)]
        refined_proposals = utils.apply_box_deltas(
            proposals,
            roi_bbox_specific * self.config.BBOX_STD_DEV).astype(
            np.int32)
        modellib.log("refined_proposals", refined_proposals)

        # Show positive proposals
        # ids = np.arange(roi_boxes.shape[0])  # Display all
        limit = 5
        ids = np.random.randint(0, len(roi_positive_ixs),
                                limit)  # Display random sample
        captions = ["{} {:.3f}".format(self.dataset_classes[c],
                                       s) if c > 0 else ""
                    for c, s in
                    zip(roi_class_ids[roi_positive_ixs][ids],
                        roi_scores[roi_positive_ixs][ids])]
        visualise.draw_boxes(image,
                             boxes=proposals[roi_positive_ixs][ids],
                             refined_boxes=
                             refined_proposals[roi_positive_ixs][ids],
                             visibilities=np.where(
                                 roi_class_ids[roi_positive_ixs][
                                     ids] > 0, 1, 0),
                             captions=captions,
                             title="ROIs After Refinement",
                             ax=self.get_ax())

        # Remove boxes classified as background
        keep = np.where(roi_class_ids > 0)[0]
        print("Keep {} detections:\n{}".format(keep.shape[0], keep))

        # Remove low confidence detections
        keep = np.intersect1d(keep, np.where(
            roi_scores >= self.config.DETECTION_MIN_CONFIDENCE)[
            0])
        print("Remove boxes below {} confidence. Keep {}:\n{}".format(
            self.config.DETECTION_MIN_CONFIDENCE, keep.shape[0],
            keep))

        # Apply per-class non-max suppression
        pre_nms_boxes = refined_proposals[keep]
        pre_nms_scores = roi_scores[keep]
        pre_nms_class_ids = roi_class_ids[keep]

        nms_keep = []
        for class_id in np.unique(pre_nms_class_ids):
            # Pick detections of this class
            ixs = np.where(pre_nms_class_ids == class_id)[0]
            # Apply NMS
            class_keep = utils.non_max_suppression(pre_nms_boxes[ixs],
                                                   pre_nms_scores[ixs],
                                                   self.config.DETECTION_NMS_THRESHOLD)
            # Map indicies
            class_keep = keep[ixs[class_keep]]
            nms_keep = np.union1d(nms_keep, class_keep)
            print("{:22}: {} -> {}".format(
                self.dataset_classes[class_id][:20],
                keep[ixs], class_keep))

        keep = np.intersect1d(keep, nms_keep).astype(np.int32)
        print("\nKept after per-class NMS: {}\n{}".format(keep.shape[0],
                                                          keep))

        # Show final detections
        ixs = np.arange(len(keep))  # Display all
        # ixs = np.random.randint(0, len(keep), 10)  # Display random sample
        captions = ["{} {:.3f}".format(self.dataset_classes[c],
                                       s) if c > 0 else ""
                    for c, s in zip(roi_class_ids[keep][ixs],
                                    roi_scores[keep][ixs])]
        visualise.draw_boxes(
            image, boxes=proposals[keep][ixs],
            refined_boxes=refined_proposals[keep][ixs],
            visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
            captions=captions, title="Detections after NMS",
            ax=self.get_ax())

    def generating_masks(self, model, image):
        # Get predictions of mask head
        mrcnn = model.run_graph([image], [
            ("detections", model.keras_model.get_layer("mrcnn_detection").output),
            ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        ])

        # Get detection class IDs. Trim zero padding.
        det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
        det_count = np.where(det_class_ids == 0)[0][0]
        det_class_ids = det_class_ids[:det_count]

        print("{} detections: {}".format(
            det_count, np.array(self.dataset_classes)[det_class_ids]))

        # Masks
        det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
        det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c]
                                      for i, c in enumerate(det_class_ids)])
        det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
                              for i, m in enumerate(det_mask_specific)])
        modellib.log("det_mask_specific", det_mask_specific)
        modellib.log("det_masks", det_masks)

        visualise.display_images(det_mask_specific[:4] * 255, cmap="Blues", interpolation="none")

        visualise.display_images(det_masks[:4] * 255, cmap="Blues", interpolation="none")

        # Get activations of a few sample layers
        activations = model.run_graph([image], [
            ("input_image",        tf.identity(model.keras_model.get_layer("input_image").output)),
            ("res4w_out",          model.keras_model.get_layer("res4w_out").output),  # for resnet100
            ("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),
            ("roi",                model.keras_model.get_layer("ROI").output),
        ])

        # Input image (normalized)
        _ = plt.imshow(modellib.unmold_image(activations["input_image"][0],self.config))

        # Backbone feature map
        visualise.display_images(np.transpose(activations["res4w_out"][0, :, :, :4], [2, 0, 1]))

        # Histograms of RPN bounding box deltas
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 4, 1)
        plt.title("dy")
        _ = plt.hist(activations["rpn_bbox"][0,:,0], 50)
        plt.subplot(1, 4, 2)
        plt.title("dx")
        _ = plt.hist(activations["rpn_bbox"][0,:,1], 50)
        plt.subplot(1, 4, 3)
        plt.title("dw")
        _ = plt.hist(activations["rpn_bbox"][0,:,2], 50)
        plt.subplot(1, 4, 4)
        plt.title("dh")
        _ = plt.hist(activations["rpn_bbox"][0,:,3], 50)

        # Distribution of y, x coordinates of generated proposals
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("y1, x1")
        plt.scatter(activations["roi"][0,:,0], activations["roi"][0,:,1])
        plt.subplot(1, 2, 2)
        plt.title("y2, x2")
        plt.scatter(activations["roi"][0,:,2], activations["roi"][0,:,3])
        plt.show()
