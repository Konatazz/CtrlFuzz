import os
import numpy as np
import cv2  
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from .models.diff_model import diff_model
import click

# python -m src.infer_seg

@click.command()
@click.option("--loadDir", "loadDir", type=str, default="models/models_res_res", help="Location of the models to load in.", required=True)
@click.option("--loadFile", "loadFile", type=str, default="model_438e_550000s.pkl", help="Name of the .pkl model file to load in.", required=True)
@click.option("--loadDefFile", "loadDefFile", type=str, default="model_params_438e_550000s.json", help="Name of the .json model file to load in.", required=True)

@click.option("--num_images", "num_images", type=int, default=50, help="Number of images to generate.", required=False)
@click.option("--step_size", "step_size", type=int, default=10, help="Step size when generating. A step size of 10 with a model trained on 1000 steps takes 100 steps to generate. Lower is faster, but produces lower quality images.", required=False)

@click.option("--DDIM_scale", "DDIM_scale", type=int, default=0, help="1:DDPM  0:DDIM.", required=False)

@click.option("--device", "device", type=str, default="gpu", help="Device to put the model on. use 'gpu' or 'cpu'.", required=False)
@click.option("--guidance", "w", type=int, default=4, help="Classifier guidance scale which must be >= 0. The higher the value, the better the image quality, but the lower the image diversity.", required=False)
@click.option("--class_label", "class_label", type=int, default=647, help="0-indexed class value. Use a class value >= 0.", required=False)
# 3, 12, 647

@click.option("--corrected", "corrected", type=bool, default=False, help="True to put a limit on generation.", required=False)

@click.option("--output_dir", "output_dir", type=str, default="output_images/seg_test", help="Directory to save the output images.", required=False)

@click.option("--num_final_images", "num_final_images", type=int, default=10, help="Number of final images to generate with the same background.", required=False)
@click.option("--mask_rcnn_threshold", "mask_rcnn_threshold", type=float, default=0.1, help="Threshold for Mask-RCNN predictions.", required=False)




def infer_seg(loadDir, loadFile, loadDefFile, num_images, step_size, DDIM_scale, device, w, class_label, corrected,
              output_dir, num_final_images, mask_rcnn_threshold):
    os.makedirs(output_dir, exist_ok=True)

    # 检查并设置设备
    if device == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("GPU not available, falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Load DDPM model
    model = diff_model(3, 3, 1, 1, ["res", "res"], 100000, "cosine", 100, device, 100, 1000, 16, 0.0,
                       step_size=step_size, DDIM_scale=DDIM_scale)
    model.loadModel(loadDir, loadFile, loadDefFile)

    # Step 1: 生成一批图像
    generated_images = []
    for i in range(num_images):
        noise = model.sample_imgs(1, class_label, w, False, False, True, corrected)
        noise = noise[0]  # 获取第一张生成的图像
        img = noise.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
        img_path = os.path.join(output_dir, f"generated_image_{i}.png")
        plt.imsave(img_path, img)
        generated_images.append(img)
        print(f"Saved image {img_path}")

    # Step 2: 使用 Mask-RCNN 分割背景并去除背景中的物体
    mask_rcnn = models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    mask_rcnn.eval()

    transform = transforms.Compose([transforms.ToTensor()])

    # 随机选择一张生成图像作为背景
    background = generated_images[0]
    background_img_tensor = transform(Image.fromarray(background)).unsqueeze(0).to(device)

    with torch.no_grad():
        background_predictions = mask_rcnn(background_img_tensor)

    background_pred_masks = background_predictions[0]['masks']
    background_pred_scores = background_predictions[0]['scores']

    # 过滤掉得分较低的掩码，只保留得分高的掩码去掉物体
    background_masks = []
    for j, score in enumerate(background_pred_scores):
        if score >= mask_rcnn_threshold:
            background_masks.append(background_pred_masks[j, 0].cpu().numpy())

    if not background_masks:
        print("No valid masks found, using the original background without changes.")  # 不抛出异常，使用原始背景
        clean_background = background
    else:
        combined_background_mask = np.clip(np.sum(background_masks, axis=0), 0, 1)  # 合并所有前景掩码

        # 使用 OpenCV inpaint 修复背景，替代物体区域
        mask_for_inpaint = (combined_background_mask * 255).astype(np.uint8)  # 转换为 8-bit 掩码
        clean_background = cv2.inpaint(background, mask_for_inpaint, 3, cv2.INPAINT_TELEA)

    # Step 3: 将新生成的前景物体与去物体后的背景进行融合
    for i, img in enumerate(generated_images):
        img_tensor = transform(Image.fromarray(img)).unsqueeze(0).to(device)

        # 对生成的图像进行分割
        with torch.no_grad():
            predictions = mask_rcnn(img_tensor)

        pred_masks = predictions[0]['masks']
        pred_scores = predictions[0]['scores']

        # 保存每个分割结果用于调试
        mask_debug_path = os.path.join(output_dir, f"mask_debug_{i}.png")
        plt.imsave(mask_debug_path, pred_masks[0, 0].cpu().numpy(), cmap="gray")
        print(f"Saved mask {mask_debug_path} with score {pred_scores[0].item()}")

        # Filter masks by score threshold
        masks = []
        for j, score in enumerate(pred_scores):
            if score >= mask_rcnn_threshold:
                masks.append(pred_masks[j, 0].cpu().numpy())

        if not masks:
            print(f"No valid masks found for image {i}. Skipping foreground blending.")
            continue  # 跳过这个图像的融合

        # Step 4: 使用更少模糊的掩码并融合前景和去掉物体的背景
        mask = masks[0]  # 选择第一个有效的掩码
        mask = np.clip(mask, 0, 1)  # 将掩码值限制在 [0, 1] 之间

        # 减少掩码的模糊强度，sigma 调整为较小的值 (sigma=1.5)
        blurred_mask = gaussian_filter(mask, sigma=0.2)

        # 调整生成的前景物体和背景大小/位置匹配
        foreground = np.array(img)

        # 将生成的前景与去物体后的背景结合，使用模糊的掩码实现平滑融合
        combined_image = (
                    blurred_mask[..., None] * foreground + (1 - blurred_mask[..., None]) * clean_background).astype(
            np.uint8)

        # 保存最终融合的图像
        final_image_path = os.path.join(output_dir, f"final_image_{i}.png")
        plt.imsave(final_image_path, combined_image)
        print(f"Saved final image {final_image_path}")


if __name__ == "__main__":
    infer_seg()
