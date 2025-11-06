"""
Training logic for CrossAttentionDet.
"""
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from ..models.encoder import get_encoder
from ..models.backbone import CrossAttentionBackbone
from ..data.dataset import NpyYoloDataset, collate_fn


class Trainer:
    """
    Trainer class for CrossAttentionDet.

    Args:
        config: Configuration object
    """

    def __init__(self, config):
        self.config = config

        # Auto-detect number of classes
        config.auto_detect_num_classes()

        # Setup dataset and dataloader
        print("\n1. Setting up dataset...")
        self.train_dataset = NpyYoloDataset(
            config.image_dir,
            config.label_dir,
            mode='train',
            test_size=config.test_size,
            random_state=config.random_state
        )

        print("\n2. Initializing DataLoader...")
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        # Build model
        print("\n3. Building Multi-modal FPN backbone...")
        self.model = self.build_model()
        self.model.to(config.device)
        print(f"Model moved to {config.device}")

        # Setup optimizer
        print("\n4. Setting up optimizer...")
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )

    def build_model(self):
        """Build the CrossAttentionDet model."""
        config = self.config

        # Instantiate the Multi-modal encoder using selected backbone
        print(f"Using backbone: {config.backbone_type}")
        encoder_base = get_encoder(
            config.backbone_type,
            in_chans_rgb=config.in_chans_rgb,
            in_chans_x=config.in_chans_x
        )

        # Wrap with FPN
        backbone = CrossAttentionBackbone(encoder_base, fpn_out_channels=config.fpn_out_channels)
        print(f"Backbone created. Output channels of FPN: {backbone.out_channels}")

        # Define the anchor generator
        anchor_generator = AnchorGenerator(
            sizes=config.anchor_sizes,
            aspect_ratios=config.anchor_aspect_ratios
        )

        # Define the RoI pooling layer
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=config.roi_featmap_names,
            output_size=config.roi_output_size,
            sampling_ratio=config.roi_sampling_ratio
        )

        # Create the Faster R-CNN model
        model = FasterRCNN(
            backbone,
            num_classes=config.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            image_mean=config.image_mean,
            image_std=config.image_std
        )

        return model

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for i, (images, targets) in enumerate(self.train_loader):
            images = list(image.to(self.config.device) for image in images)
            targets = [{k: v.to(self.config.device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()

            if (i + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{self.config.num_epochs}], "
                      f"Step [{i+1}/{len(self.train_loader)}], "
                      f"Loss: {losses.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        """Full training loop."""
        print(f"\n5. Starting training loop for {self.config.num_epochs} epochs...")

        for epoch in range(self.config.num_epochs):
            self.train_epoch(epoch)

        # Save the trained model
        self.save_checkpoint()

    def save_checkpoint(self, path=None):
        """Save model checkpoint."""
        if path is None:
            path = self.config.model_path
        torch.save(self.model.state_dict(), path)
        print(f"\n--- Training Finished ---\nModel saved to {path}")


__all__ = ['Trainer']
