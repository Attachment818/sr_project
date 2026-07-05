import configparser
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.ndimage import map_coordinates

from common.common_util import pre_processing, simple_nms, remove_borders, \
    sample_keypoint_desc, remove_keypoints_by_mask
from model.super_retina import SuperRetina, SuperRetinaFPN, SuperRetinaWithSelfAttention, SuperRetinaWithMultiScaleDescriptor, SuperRetinaWithASPP, SuperRetinaWithoutPKE, SuperRetinaWithoutPKEWithAttention,SuperRetinaWithPerceptualLoss,SuperRetinaWithVesselRegularization,SuperRetinaWithVesselOnly
from model.super_retina_attention import SuperRetinaWithAttention

from PIL import Image
import os


class Predictor:
    def __init__(self, config):

        predict_config = config['PREDICT']

        device = predict_config['device']
        device = torch.device(device if torch.cuda.is_available() else "cpu")

        model_save_path = predict_config['model_save_path']
        self.nms_size = predict_config['nms_size']
        self.nms_thresh = predict_config['nms_thresh']
        self.scale = 8
        self.knn_thresh = predict_config['knn_thresh']

        self.image_width = None
        self.image_height = None

        self.model_image_width = predict_config['model_image_width']
        self.model_image_height = predict_config['model_image_height']

        # FIMD 等跨分辨率数据集：将 refer 缩放到 query 尺寸后再匹配
        self.resize_refer_to_query = predict_config.get('resize_refer_to_query', False)
        
        # Eye boundary mask (optional)
        # If provided, keypoints on eye boundary (mask == 0) will be filtered out
        self.eye_mask = None
        
        # Load eye mask from config if path is specified
        eye_mask_path = predict_config.get('eye_mask_path', None)
        if eye_mask_path is not None and os.path.exists(eye_mask_path):
            try:
                eye_mask = cv2.imread(eye_mask_path, cv2.IMREAD_GRAYSCALE)
                if eye_mask is not None:
                    eye_mask = eye_mask.astype(np.float32) / 255.0
                    self.eye_mask = eye_mask
                    print(f"Eye mask loaded from: {eye_mask_path}")
                else:
                    print(f"Warning: Failed to load eye mask from {eye_mask_path}")
            except Exception as e:
                print(f"Warning: Error loading eye mask from {eye_mask_path}: {e}")

        # 从配置中读取模型类型，默认为'original'（保持向后兼容）
        model_type = predict_config.get('model_type', 'original')
        use_attention = predict_config.get('use_attention', True)
        attention_type = predict_config.get('attention_type', 'CBAM')
        use_self_attention = predict_config.get('use_self_attention', True)
        attention_reduction = predict_config.get('attention_reduction', 4)
        use_multi_scale_desc = predict_config.get('use_multi_scale_desc', True)
        use_aspp = predict_config.get('use_aspp', True)
        aspp_rates = predict_config.get('aspp_rates', [1, 2, 4, 8])
        
        checkpoint = torch.load(model_save_path, map_location=device)
        
        # 根据配置选择使用哪个模型
        if model_type == 'without_pke':
            # 使用不带PKE模块的模型
            model = SuperRetinaWithoutPKE(
                config=None,  # 预测时不需要config
                device=device
            )
            # 使用安全加载方法，可以加载原始SuperRetina模型权重
            model.load_pretrained_weights(model_save_path, device=device, strict=False)
            print(f"Loaded SuperRetinaWithoutPKE model (PKE_learn=False)")
        elif model_type == 'without_pke_attention':
            # 使用不带PKE模块 + 带自注意力机制的模型
            use_self_attention = predict_config.get('use_self_attention', True)
            attention_reduction = predict_config.get('attention_reduction', 8)
            model = SuperRetinaWithoutPKEWithAttention(
                config=None,  # 预测时不需要config
                device=device,
                use_self_attention=use_self_attention,
                attention_reduction=attention_reduction
            )
            # 使用安全加载方法，可以加载原始SuperRetina模型权重
            model.load_pretrained_weights(model_save_path, device=device, strict=False)
            print(f"Loaded SuperRetinaWithoutPKEWithAttention model (PKE_learn=False, self_attention={use_self_attention}, reduction={attention_reduction})")
        elif model_type == 'aspp':
            # 使用带ASPP模块的模型
            model = SuperRetinaWithASPP(
                config=None,  # 预测时不需要config
                device=device,
                use_aspp=use_aspp,
                aspp_rates=aspp_rates
            )
            # 使用安全加载方法，可以加载旧模型权重
            model.load_pretrained_weights(model_save_path, device=device, strict=False)
            print(f"Loaded SuperRetinaWithASPP model (aspp={use_aspp}, rates={aspp_rates})")
        elif model_type == 'multiscale_desc':
            # 使用带多尺度描述子融合的模型
            model = SuperRetinaWithMultiScaleDescriptor(
                config=None,  # 预测时不需要config
                device=device,
                use_multi_scale_desc=use_multi_scale_desc
            )
            # 使用安全加载方法，可以加载旧模型权重
            model.load_pretrained_weights(model_save_path, device=device, strict=False)
            print(f"Loaded SuperRetinaWithMultiScaleDescriptor model (multi_scale_desc={use_multi_scale_desc})")
        elif model_type == 'attention':
            # 使用带注意力机制的模型（CBAM/Channel/Spatial）
            model = SuperRetinaWithAttention(
                config=None,  # 预测时不需要config
                device=device,
                use_attention=use_attention,
                attention_type=attention_type
            )
            # 使用安全加载方法，可以加载旧模型权重
            model.load_pretrained_weights(model_save_path, device=device, strict=False)
            print(f"Loaded SuperRetinaWithAttention model (attention={use_attention}, type={attention_type})")
        elif model_type == 'self_attention':
            # 使用带自注意力机制的模型
            model = SuperRetinaWithSelfAttention(
                config=None,  # 预测时不需要config
                device=device,
                use_self_attention=use_self_attention,
                attention_reduction=attention_reduction
            )
            # 使用安全加载方法，可以加载旧模型权重
            model.load_pretrained_weights(model_save_path, device=device, strict=False)
            print(f"Loaded SuperRetinaWithSelfAttention model (self_attention={use_self_attention}, reduction={attention_reduction})")
        
        elif model_type == 'fpn':
            # 使用带 FPN 解码头的模型（与 train_with_fpn.py 对应）
            model = SuperRetinaFPN()
            if 'net' in checkpoint:
                pretrained_dict = checkpoint['net']
            else:
                pretrained_dict = checkpoint
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in pretrained_dict.items()
                             if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded SuperRetinaFPN model from {model_save_path} "
                  f"(matched {len(filtered_dict)}/{len(pretrained_dict)} tensors)")
        elif model_type == 'with_perceptual_loss':
            model = SuperRetinaWithPerceptualLoss(
                config=None, device=device
            )
            model.load_pretrained_weights(model_save_path, device=device, strict=False)
            print(f"Loaded SuperRetinaWithPerceptualLoss model (perceptual_weight={model.perceptual_weight})")
        elif model_type == 'with_vessel_regularization':
            # 新增：支持我们刚刚添加的带 vessel regularization 的优化模型
            model = SuperRetinaWithVesselRegularization(
                config=None,  # 预测时不需要 config
                device=device
            )
            # 使用我们已经在类中实现的安全加载方法
            model.load_pretrained_weights(model_save_path, device=device, strict=False)
            print(f"✅ Loaded SuperRetinaWithVesselRegularization model (vessel_weight={getattr(model, 'vessel_weight', 'N/A')})")
        elif model_type == 'with_vessel_only':
            model = SuperRetinaWithVesselOnly(
                config=None,
                device=device
            )
            model.load_pretrained_weights(model_save_path, device=device, strict=False)
            print(f"✅ Loaded SuperRetinaWithVesselOnly model (vessel_weight={getattr(model, 'vessel_weight', 'N/A')})")
        else:
            # 使用原始模型（默认，保持向后兼容），并兼容带 FPN/注意力的权重文件
            model = SuperRetina()
            if 'net' in checkpoint:
                pretrained_dict = checkpoint['net']
            else:
                pretrained_dict = checkpoint
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in pretrained_dict.items()
                             if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded original SuperRetina model from {model_save_path} "
                  f"(matched {len(filtered_dict)}/{len(pretrained_dict)} tensors; "
                  f"ignored extra keys such as FPN/attention modules if present)")
        
        model.to(device)
        model.eval()
        self.device = device
        self.model = model
        self.knn_matcher = cv2.BFMatcher(cv2.NORM_L2)

        self.trasformer = transforms.Compose([
            transforms.Resize((self.model_image_height, self.model_image_width)),
            transforms.ToTensor(),

        ])
    
    def set_eye_mask(self, eye_mask):
        """
        Set the eye boundary mask for filtering keypoints
        :param eye_mask: numpy array or torch.Tensor of shape (H, W)
                         Values > 0.5 indicate valid regions (keep keypoints)
                         Values <= 0.5 indicate eye boundary (remove keypoints)
        """
        self.eye_mask = eye_mask

    def image_read(self, query_path, refer_path, query_is_image=False):
        if query_is_image:
            query_image = query_path
        else:
            query_image = cv2.imread(query_path, cv2.IMREAD_COLOR)
            # green channel
            query_image = query_image[:, :, 1]
            query_image = pre_processing(query_image)
        refer_image = cv2.imread(refer_path, cv2.IMREAD_COLOR)

        if query_image.shape[:2] != refer_image.shape[:2]:
            if self.resize_refer_to_query:
                refer_image = cv2.resize(
                    refer_image,
                    (query_image.shape[1], query_image.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                raise AssertionError(
                    f"Image size mismatch: query {query_image.shape[:2]} vs refer {refer_image.shape[:2]}. "
                    "Set resize_refer_to_query: true for FIMD."
                )
        self.image_height, self.image_width = query_image.shape[:2]

        refer_image = refer_image[:, :, 1]
        refer_image = pre_processing(refer_image)

        query_image = (query_image * 255).astype(np.uint8)
        refer_image = (refer_image * 255).astype(np.uint8)

        return query_image, refer_image

    def draw_result(self, query_image, refer_image, cv_kpts_query, cv_kpts_refer, matches, status):
        def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
            # initialize the output visualization image
            (hA, wA) = imageA.shape[:2]
            (hB, wB) = imageB.shape[:2]
            vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
            if len(imageA.shape) == 2:
                imageA = cv2.cvtColor(imageA, cv2.COLOR_GRAY2RGB)
                imageB = cv2.cvtColor(imageB, cv2.COLOR_GRAY2RGB)

            vis[0:hA, 0:wA] = imageA
            vis[0:hB, wA:] = imageB

            # loop over the matches
            for (match, _), s in zip(matches, status):
                trainIdx, queryIdx = match.trainIdx, match.queryIdx
                # only process the match if the keypoint was successfully
                # matched
                if s == 1:
                    # draw the match
                    ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
                    ptB = (int(kpsB[trainIdx].pt[0]) + wA, int(kpsB[trainIdx].pt[1]))
                    cv2.line(vis, ptA, ptB, (0, 255, 0), 2)

                # return the visualization
            return vis

        query_np = np.array([kp.pt for kp in cv_kpts_query])
        refer_np = np.array([kp.pt for kp in cv_kpts_refer])
        refer_np[:, 0] += query_image.shape[1]
        matched_image = drawMatches(query_image, refer_image, cv_kpts_query, cv_kpts_refer, matches, status)
        plt.figure(dpi=300)
        plt.scatter(query_np[:, 0], query_np[:, 1], s=1, c='r')
        plt.scatter(refer_np[:, 0], refer_np[:, 1], s=1, c='r')
        plt.axis('off')
        plt.title('Match Result, #goodMatch: {}'.format(status.sum()))
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.show()
        plt.close()

    def model_run_pair(self, query_tensor, refer_tensor):
        inputs = torch.cat((query_tensor.unsqueeze(0), refer_tensor.unsqueeze(0)))
        inputs = inputs.to(self.device)

        with torch.no_grad():
            detector_pred, descriptor_pred = self.model(inputs)

        scores = simple_nms(detector_pred, self.nms_size)

        b, _, h, w = detector_pred.shape
        scores = scores.reshape(-1, h, w)

        keypoints = [
            torch.nonzero(s > self.nms_thresh)
            for s in scores]

        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, 4, h, w)
            for k, s in zip(keypoints, scores)]))

        # Discard keypoints on eye boundary based on mask
        if self.eye_mask is not None:
            keypoints, scores = list(zip(*[
                remove_keypoints_by_mask(k, s, self.eye_mask, h, w)
                for k, s in zip(keypoints, scores)]))

        keypoints = [torch.flip(k, [1]).float().data for k in keypoints]

        descriptors = [sample_keypoint_desc(k[None], d[None], 8)[0].cpu()
                       for k, d in zip(keypoints, descriptor_pred)]
        keypoints = [k.cpu() for k in keypoints]
        return keypoints, descriptors

    def match(self, query_path, refer_path, show=False, query_is_image=False):
        query_image, refer_image = self.image_read(query_path, refer_path, query_is_image)
        query_tensor = self.trasformer(Image.fromarray(query_image))
        refer_tensor = self.trasformer(Image.fromarray(refer_image))

        keypoints, descriptors = self.model_run_pair(query_tensor, refer_tensor)

        query_keypoints, refer_keypoints = keypoints[0], keypoints[1]
        query_desc, refer_desc = descriptors[0].permute(1, 0).numpy(), descriptors[1].permute(1, 0).numpy()

        # mapping keypoints to scaled keypoints 把关键点映射到原图尺寸上
        cv_kpts_query = [cv2.KeyPoint(int(i[0] / self.model_image_width * self.image_width),
                                      int(i[1] / self.model_image_height * self.image_height), 30)
                         for i in query_keypoints]
        cv_kpts_refer = [cv2.KeyPoint(int(i[0] / self.model_image_width * self.image_width),
                                      int(i[1] / self.model_image_height * self.image_height), 30)
                         for i in refer_keypoints]

        goodMatch = []
        status = []
        matches = []
        try:
            matches = self.knn_matcher.knnMatch(query_desc, refer_desc, k=2)
            for m, n in matches:
                if m.distance < self.knn_thresh * n.distance:
                    goodMatch.append(m)
                    status.append(True)
                else:
                    status.append(False)
        except Exception:
            pass

        if show:
            self.draw_result(query_image, refer_image, cv_kpts_query, cv_kpts_refer, matches, np.array(status))
        return goodMatch, cv_kpts_query, cv_kpts_refer, query_image, refer_image

    ### 单应性矩阵是在推理时进行两张图的估计变换，而合成映射是在训练阶段实现PKE模块中的几何一致性检查和描述子的鲁棒性增强
    def compute_homography(self, query_path, refer_path, query_is_image=False):
        goodMatch, cv_kpts_query, cv_kpts_refer, raw_query_image, raw_refer_image = \
            self.match(query_path, refer_path, query_is_image=query_is_image)
        H_m = None
        inliers_num_rate = 0

        if len(goodMatch) >= 4:
            src_pts = [cv_kpts_query[m.queryIdx].pt for m in goodMatch]
            src_pts = np.float32(src_pts).reshape(-1, 1, 2)
            dst_pts = [cv_kpts_refer[m.trainIdx].pt for m in goodMatch]
            dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

            H_m, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
            # H_m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0, 
            #                               maxIters=2000, 
            #                               confidence=0.995)

            # src_pts = src_pts[mask.ravel() == 1]
            # dst_pts = dst_pts[mask.ravel() == 1]

            goodMatch = np.array(goodMatch)[mask.ravel() == 1]
            inliers_num_rate = mask.sum() / len(mask.ravel()) ### ravel方法将多维数组展平为1维

        return H_m, inliers_num_rate, raw_query_image, raw_refer_image

    ### 用于文件内单对测试，实际上test_on_FIRE没有使用
    def align_image_pair(self, query_path, refer_path, show=False):
        H_m, inliers_num_rate, raw_query_image, raw_refer_image = self.compute_homography(query_path, refer_path)

        if H_m is not None:
            h, w = self.image_height, self.image_width
            ### 透视变换对齐
            query_align = cv2.warpPerspective(raw_query_image, 
                                              H_m, 
                                              (w, h), # 变换尺寸 
                                              borderMode=cv2.BORDER_CONSTANT, ### 边界值之外怎么填充，这里是固定值
                                              borderValue=(0)) ### 固定值为0

            merged = np.zeros((h, w, 3), dtype=np.uint8)
            
            ### 转换为灰度图，只有亮度，等下放到通道里变成通道颜色
            if len(query_align.shape) == 3:
                query_align = cv2.cvtColor(query_align, cv2.COLOR_BGR2GRAY)
            if len(raw_refer_image.shape) == 3:
                refer_gray = cv2.cvtColor(raw_refer_image, cv2.COLOR_BGR2GRAY)
            else:
                refer_gray = raw_refer_image
            ### RGB图通道顺序固定是B_G_R，查询图放B，参考图放G，最终图像青色部分就是重合对齐部分
            merged[:, :, 0] = query_align
            merged[:, :, 1] = refer_gray

            if show:
                plt.figure(dpi=200)
                plt.imshow(merged)
                plt.axis('off')
                plt.title('Registration Result')
                plt.show()
                plt.close()
            return merged

        print("Matched Failed!")

    def compute_quadratic_matrix(self, landmarks1, landmarks2):
        """
        计算二阶多项式变换矩阵
        
        使用最小二乘法计算二阶多项式系数，变换公式为：
        x' = a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2 + a6
        y' = a7*x + a8*y + a9*x*y + a10*x^2 + a11*y^2 + a12
        
        Parameters:
        - landmarks1: query 图像中的点坐标列表 [(x, y), ...]
        - landmarks2: refer 图像中对应的点坐标列表 [(x, y), ...]
        
        Returns:
        - coefficients: 12个系数的数组 [a1, a2, ..., a12]
        """
        if len(landmarks1) != len(landmarks2) or len(landmarks1) < 6:
            raise ValueError("Both landmarks should have the same number of points, and at least 6 points are required.")
        
        A = []
        B = []
        
        for (x, y), (x_prime, y_prime) in zip(landmarks1, landmarks2):
            # For x'
            A.append([x, y, x*y, x*x, y*y, 1, 0, 0, 0, 0, 0, 0])
            # For y'
            A.append([0, 0, 0, 0, 0, 0, x, y, x*y, x*x, y*y, 1])
            
            B.append(x_prime)
            B.append(y_prime)
        
        A = np.array(A)
        B = np.array(B)
        
        # Solve the linear system
        coefficients, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        
        return coefficients

    def transform_points_quadratic(self, points, coefficients):
        """
        使用二阶多项式系数变换点坐标
        
        Parameters:
        - points: 点坐标列表 [(x, y), ...]
        - coefficients: 12个系数的数组 [a1, a2, ..., a12]
        
        Returns:
        - transformed_points: 变换后的点坐标列表 [(x', y'), ...]
        """
        if len(coefficients) != 12:
            raise ValueError("Coefficients should have a shape of (12,).")
        
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 = coefficients
        
        transformed_points = []
        for x, y in points:
            x_prime = a1*x + a2*y + a3*x*y + a4*x**2 + a5*y**2 + a6
            y_prime = a7*x + a8*y + a9*x*y + a10*x**2 + a11*y**2 + a12
            transformed_points.append((x_prime, y_prime))
        
        return transformed_points

    def warp_image_quadratic(self, image, coefficients):
        """
        按前向多项式系数在输入像素格上重采样（非配准常用路径）。
        将 query 配准到 refer 画布请使用：
          coeffs_inv = compute_quadratic_matrix(dst_pts, src_pts)
          warp_image_quadratic_inverse_map(query, coeffs_inv, out_h=H_refer, out_w=W_refer)
        """
        if len(coefficients) != 12:
            raise ValueError("Coefficients should have a shape of (12,).")
        
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 = coefficients
        
        # Check if the image is grayscale or colored
        if len(image.shape) == 2:
            height, width = image.shape
            channels = 1
            # 为灰度图添加通道维度以便统一处理
            image = np.expand_dims(image, axis=2)  # shape: (height, width, 1)
            output = np.zeros((height, width, 1))
        else:
            height, width, channels = image.shape
            output = np.zeros((height, width, channels))
        
        # Generate the coordinates
        coordinates = np.indices((height, width))
        x_coords = coordinates[1]
        y_coords = coordinates[0]
        
        # Compute new x' and y' for every x and y
        x_prime = a1*x_coords + a2*y_coords + a3*x_coords*y_coords + a4*x_coords**2 + a5*y_coords**2 + a6
        y_prime = a7*x_coords + a8*y_coords + a9*x_coords*y_coords + a10*x_coords**2 + a11*y_coords**2 + a12
        
        # Map the old image pixels to the new deformed positions
        for c in range(channels):  # for each channel
            output[:, :, c] = map_coordinates(image[:, :, c], [y_prime, x_prime], order=1, mode='constant', cval=0.0)
        
        if channels == 1:
            return output[:, :, 0]  # return as 2D grayscale image
        else:
            return output

    def warp_image_quadratic_inverse_map(self, image, coefficients, out_h, out_w, shift_x=0.0, shift_y=0.0, stripe_h=512):
        """
        使用二阶多项式系数进行**反向映射**warp：
        - 对于输出图像坐标系中的每个像素 (x_out, y_out)，先换算到多项式坐标系：
            x = x_out - shift_x
            y = y_out - shift_y
        - 然后用 coefficients 计算其在输入图像中的采样坐标 (x_in, y_in)
        - 最后对输入图像做插值采样得到输出像素值

        注意：
        - coefficients 必须表示 (x, y) -> (x_in, y_in) 的映射（也就是“输出/目标坐标 -> 输入/源坐标”）。
        - 这是图像warp最稳定的方式（不会出现forward splat导致的网格空洞）。
        """
        if len(coefficients) != 12:
            raise ValueError("Coefficients should have a shape of (12,).")

        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 = coefficients

        # Normalize input to HxWxC
        if len(image.shape) == 2:
            in_h, in_w = image.shape
            channels = 1
            image_c = np.expand_dims(image, axis=2)
        else:
            in_h, in_w, channels = image.shape
            image_c = image

        # Allocate output
        output = np.zeros((out_h, out_w, channels), dtype=np.float32)

        # Stripe processing to limit peak memory
        for y0 in range(0, out_h, stripe_h):
            y1 = min(out_h, y0 + stripe_h)
            stripe_h_eff = y1 - y0

            # Output coords grid (y, x)
            y_coords, x_coords = np.mgrid[y0:y1, 0:out_w]
            x = x_coords.astype(np.float32) - float(shift_x)
            y = y_coords.astype(np.float32) - float(shift_y)

            # Map to input sample coords
            x_in = a1*x + a2*y + a3*x*y + a4*x**2 + a5*y**2 + a6
            y_in = a7*x + a8*y + a9*x*y + a10*x**2 + a11*y**2 + a12

            # Sample each channel
            for c in range(channels):
                output[y0:y1, :, c] = map_coordinates(
                    image_c[:, :, c],
                    [y_in, x_in],
                    order=1,
                    mode='constant',
                    cval=0.0
                )

        if channels == 1:
            return output[:, :, 0]
        return output

    def compute_quadratic(self, query_path, refer_path, query_is_image=False):
        """
        计算二阶多项式变换矩阵（类似 compute_homography）
        
        Parameters:
        - query_path: query 图像路径或图像数组
        - refer_path: refer 图像路径
        - query_is_image: 如果 query_path 是图像数组，设为 True
        
        Returns:
        - coefficients: query -> refer 的前向系数 (12)，失败为 None
        - coefficients_inv: refer -> query，供 warp_image_quadratic_inverse_map 做图像 warp；失败为 None
        - inliers_num_rate: 内点比例
        - raw_query_image: query 图像
        - raw_refer_image: refer 图像
        """
        goodMatch, cv_kpts_query, cv_kpts_refer, raw_query_image, raw_refer_image = \
            self.match(query_path, refer_path, query_is_image=query_is_image)
        
        coefficients = None
        inliers_num_rate = 0
        
        coefficients_inv = None
        if len(goodMatch) >= 6:  # 二阶多项式至少需要6个点
            src_pts = [cv_kpts_query[m.queryIdx].pt for m in goodMatch]
            dst_pts = [cv_kpts_refer[m.trainIdx].pt for m in goodMatch]
            
            try:
                coefficients = self.compute_quadratic_matrix(src_pts, dst_pts)
                # refer -> query：供 warp_image_quadratic_inverse_map（输出为 refer 画布上每点对应 query 采样坐标）
                coefficients_inv = self.compute_quadratic_matrix(dst_pts, src_pts)
                inliers_num_rate = 1.0  # 对于最小二乘法，所有点都参与计算
            except Exception as e:
                print(f"Warning: Error computing quadratic matrix: {e}")
                coefficients = None
                coefficients_inv = None
                inliers_num_rate = 0
        
        return coefficients, coefficients_inv, inliers_num_rate, raw_query_image, raw_refer_image

    def align_image_pair_quadratic(self, query_path, refer_path, show=False):
        """
        使用二阶多项式变换进行图像配准（类似 align_image_pair）
        
        Parameters:
        - query_path: query 图像路径或图像数组
        - refer_path: refer 图像路径
        - show: 是否显示配准结果
        
        Returns:
        - merged: 配准后的合并图像，如果失败返回 None
        """
        coefficients, coefficients_inv, inliers_num_rate, raw_query_image, raw_refer_image = \
            self.compute_quadratic(query_path, refer_path)
        
        if coefficients is not None and coefficients_inv is not None:
            h, w = self.image_height, self.image_width
            
            # 反向映射：refer 画布上每点对应 query 中采样位置（与 backend algorithm_processor 一致）
            query_align = self.warp_image_quadratic_inverse_map(
                raw_query_image, coefficients_inv, out_h=h, out_w=w
            )
            query_align = np.clip(query_align, 0, 255).astype(np.uint8)
            
            merged = np.zeros((h, w, 3), dtype=np.uint8)
            
            if len(query_align.shape) == 3:
                query_align = cv2.cvtColor(query_align, cv2.COLOR_BGR2GRAY)
            if len(raw_refer_image.shape) == 3:
                refer_gray = cv2.cvtColor(raw_refer_image, cv2.COLOR_BGR2GRAY)
            else:
                refer_gray = raw_refer_image
            
            merged[:, :, 0] = query_align
            merged[:, :, 1] = refer_gray
            
            if show:
                plt.figure(dpi=200)
                plt.imshow(merged)
                plt.axis('off')
                plt.title('Registration Result (Quadratic)')
                plt.show()
                plt.close()
            return merged
        
        print("Quadratic Registration Failed!")
        return None

    def model_run_one_image(self, image_path, save_path=None):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = image[:, :, 1]
        self.image_height, self.image_width = image.shape[:2]

        image = pre_processing(image)
        image_tensor = self.trasformer(Image.fromarray(image))
        inputs = image_tensor.unsqueeze(0)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            detector_pred, descriptor_pred = self.model(inputs)

        scores = simple_nms(detector_pred, self.nms_size)

        b, _, h, w = detector_pred.shape
        scores = scores.reshape(-1, h, w)

        keypoints = [
            torch.nonzero(s > self.nms_thresh)
            for s in scores]

        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, 4, h, w)
            for k, s in zip(keypoints, scores)]))

        # Discard keypoints on eye boundary based on mask
        if self.eye_mask is not None:
            keypoints, scores = list(zip(*[
                remove_keypoints_by_mask(k, s, self.eye_mask, h, w)
                for k, s in zip(keypoints, scores)]))

        keypoints = [torch.flip(k, [1]).float().data for k in keypoints]

        descriptors = [sample_keypoint_desc(k[None], d[None], 8)[0].cpu()
                       for k, d in zip(keypoints, descriptor_pred)]
        keypoints = [k.cpu() for k in keypoints]

        if save_path is not None:
            save_info = {'kp': keypoints[0].cpu(), 'desc': descriptors[0].cpu()}
            torch.save(save_info, save_path)

        return keypoints[0], descriptors[0]

    def homography_from_tensor(self, query_info, refer_info):
        query_keypoints, query_desc = query_info['kp'], query_info['desc']
        refer_keypoints, refer_desc = refer_info['kp'], refer_info['desc']

        query_desc = query_desc.permute(1, 0).numpy()
        refer_desc = refer_desc.permute(1, 0).numpy()
        cv_kpts_query = [cv2.KeyPoint(int(i[0] / self.model_image_width * self.image_width),
                                      int(i[1] / self.model_image_height * self.image_height), 30)
                         for i in query_keypoints]
        cv_kpts_refer = [cv2.KeyPoint(int(i[0] / self.model_image_width * self.image_width),
                                      int(i[1] / self.model_image_height * self.image_height), 30)
                         for i in refer_keypoints]

        goodMatch = []
        status = []
        try:
            matches = self.knn_matcher.knnMatch(query_desc, refer_desc, k=2)
            for m, n in matches:
                if m.distance < self.knn_thresh * n.distance:
                    goodMatch.append(m)
                    status.append(True)
                else:
                    status.append(False)
        except Exception:
            pass

        H_m = None
        inliers_num = 0

        if len(goodMatch) >= 4:
            src_pts = [cv_kpts_query[m.queryIdx].pt for m in goodMatch]
            src_pts = np.float32(src_pts).reshape(-1, 1, 2)
            dst_pts = [cv_kpts_refer[m.trainIdx].pt for m in goodMatch]
            dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

            H_m, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
            # H_m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0, 
            #                               maxIters=2000, 
            #                               confidence=0.995)

            # src_pts = src_pts[mask.ravel() == 1]
            # dst_pts = dst_pts[mask.ravel() == 1]

            goodMatch = np.array(goodMatch)[mask.ravel() == 1]
            inliers_num = mask.sum()

        return H_m, inliers_num

    def check_inverse_consistency(self, query_keypoints, refer_keypoints, query_desc, refer_desc, iccl=3.0):
        """
        检查匹配点的逆一致性（Inverse Consistency Check）
        
        对于 query 中的每个关键点，找到它在 refer 中的最佳匹配点，
        然后检查从 refer 匹配回 query 是否回到原始点附近。
        只保留满足逆一致性条件的匹配点对。
        
        Parameters:
        - query_keypoints: query 图像的关键点列表 (cv2.KeyPoint 对象列表)
        - refer_keypoints: refer 图像的关键点列表 (cv2.KeyPoint 对象列表)
        - query_desc: query 图像的描述符 (numpy array, shape: [N, D])
        - refer_desc: refer 图像的描述符 (numpy array, shape: [M, D])
        - iccl: 逆一致性标准限制（Inverse Consistency Criteria Limit），默认 3.0 像素
        
        Returns:
        - filtered_matches: 过滤后的匹配点列表 (cv2.DMatch 对象列表)
        - query_pts: query 图像中满足条件的点坐标列表 [(x, y), ...]
        - refer_pts: refer 图像中满足条件的点坐标列表 [(x, y), ...]
        """
        if len(query_keypoints) == 0 or len(refer_keypoints) == 0:
            return [], [], []
        
        # 第一步：从 query 到 refer 的匹配
        matches_q2r = []
        try:
            matches = self.knn_matcher.knnMatch(query_desc, refer_desc, k=2)
            for m, n in matches:
                if m.distance < self.knn_thresh * n.distance:
                    matches_q2r.append(m)
        except Exception:
            return [], [], []
        
        if len(matches_q2r) == 0:
            return [], [], []
        
        # 第二步：从 refer 到 query 的匹配（反向匹配）
        matches_r2q = []
        try:
            matches = self.knn_matcher.knnMatch(refer_desc, query_desc, k=2)
            for m, n in matches:
                if m.distance < self.knn_thresh * n.distance:
                    matches_r2q.append(m)
        except Exception:
            return [], [], []
        
        if len(matches_r2q) == 0:
            return [], [], []
        
        # 创建反向匹配的查找表：refer_idx -> query_idx
        r2q_map = {}
        for match in matches_r2q:
            refer_idx = match.queryIdx  # 在 refer 中的索引
            query_idx = match.trainIdx  # 在 query 中的索引
            if refer_idx not in r2q_map:
                r2q_map[refer_idx] = query_idx
        
        # 第三步：检查逆一致性
        filtered_matches = []
        query_pts = []
        refer_pts = []
        
        for match in matches_q2r:
            query_idx = match.queryIdx
            refer_idx = match.trainIdx
            
            # 检查是否存在反向匹配
            if refer_idx in r2q_map:
                back_query_idx = r2q_map[refer_idx]
                
                # 如果反向匹配回到同一个 query 点，说明是双向一致的
                if back_query_idx == query_idx:
                    query_pt = query_keypoints[query_idx].pt
                    refer_pt = refer_keypoints[refer_idx].pt
                    filtered_matches.append(match)
                    query_pts.append(query_pt)
                    refer_pts.append(refer_pt)
                else:
                    # 检查反向匹配的点是否在 iccl 范围内
                    original_query_pt = query_keypoints[query_idx].pt
                    back_query_pt = query_keypoints[back_query_idx].pt
                    distance = np.sqrt((original_query_pt[0] - back_query_pt[0])**2 + 
                                     (original_query_pt[1] - back_query_pt[1])**2)
                    
                    if distance <= iccl:
                        query_pt = query_keypoints[query_idx].pt
                        refer_pt = refer_keypoints[refer_idx].pt
                        filtered_matches.append(match)
                        query_pts.append(query_pt)
                        refer_pts.append(refer_pt)
        
        return filtered_matches, query_pts, refer_pts

    def filter_outliers(self, query_pts, refer_pts, criteria='homography', threshold=20.0):
        """
        基于变换误差过滤异常值
        
        使用单应性或仿射变换估计，然后过滤误差大于阈值的匹配点对。
        
        Parameters:
        - query_pts: query 图像中的点坐标列表 [(x, y), ...]
        - refer_pts: refer 图像中的点坐标列表 [(x, y), ...]
        - criteria: 使用的变换类型，'homography' 或 'affine'，默认 'homography'
        - threshold: 误差阈值（像素），默认 20.0
        
        Returns:
        - filtered_query_pts: 过滤后的 query 点坐标列表
        - filtered_refer_pts: 过滤后的 refer 点坐标列表
        - transformation_matrix: 估计的变换矩阵（单应性或仿射）
        """
        if len(query_pts) < 4 or len(refer_pts) < 4:
            return query_pts, refer_pts, None
        
        if len(query_pts) != len(refer_pts):
            return query_pts, refer_pts, None
        
        query_pts_array = np.float32(query_pts).reshape(-1, 1, 2)
        refer_pts_array = np.float32(refer_pts).reshape(-1, 1, 2)
        
        try:
            if criteria == 'homography':
                # 使用单应性变换
                transformation_matrix, mask = cv2.findHomography(
                    query_pts_array, refer_pts_array, 
                    cv2.RANSAC, 
                    ransacReprojThreshold=threshold
                )
            else:
                # 使用仿射变换
                transformation_matrix, mask = cv2.estimateAffinePartial2D(
                    query_pts_array, refer_pts_array,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=threshold
                )
                # 将仿射矩阵转换为 3x3 格式以便统一处理
                if transformation_matrix is not None:
                    affine_3x3 = np.vstack([transformation_matrix, [0, 0, 1]])
                    transformation_matrix = affine_3x3
            
            if transformation_matrix is None:
                return query_pts, refer_pts, None
            
            # 使用 mask 过滤点
            if mask is not None:
                mask = mask.ravel() == 1
                filtered_query_pts = [query_pts[i] for i in range(len(query_pts)) if mask[i]]
                filtered_refer_pts = [refer_pts[i] for i in range(len(refer_pts)) if mask[i]]
            else:
                # 如果没有 mask，手动计算误差并过滤
                filtered_query_pts = []
                filtered_refer_pts = []
                
                for i, (q_pt, r_pt) in enumerate(zip(query_pts, refer_pts)):
                    # 变换 query 点到 refer 空间
                    if criteria == 'homography':
                        pt_homo = np.array([q_pt[0], q_pt[1], 1.0])
                        transformed_pt = transformation_matrix @ pt_homo
                        transformed_pt = transformed_pt[:2] / transformed_pt[2]
                    else:
                        pt_homo = np.array([q_pt[0], q_pt[1], 1.0])
                        transformed_pt = transformation_matrix[:2] @ pt_homo
                    
                    # 计算误差
                    error = np.sqrt((transformed_pt[0] - r_pt[0])**2 + (transformed_pt[1] - r_pt[1])**2)
                    
                    if error <= threshold:
                        filtered_query_pts.append(q_pt)
                        filtered_refer_pts.append(r_pt)
            
            return filtered_query_pts, filtered_refer_pts, transformation_matrix
            
        except Exception as e:
            print(f"Warning: Error in filter_outliers: {e}")
            return query_pts, refer_pts, None

    def match_with_consistency_check(self, query_path, refer_path, query_is_image=False, 
                                     use_inverse_consistency=True, iccl=3.0,
                                     use_outlier_filter=True, outlier_criteria='homography', outlier_threshold=20.0):
        """
        带逆一致性检查和异常值过滤的匹配方法
        
        Parameters:
        - query_path: query 图像路径或图像数组
        - refer_path: refer 图像路径
        - query_is_image: 如果 query_path 是图像数组，设为 True
        - use_inverse_consistency: 是否使用逆一致性检查，默认 True
        - iccl: 逆一致性标准限制，默认 3.0 像素
        - use_outlier_filter: 是否使用异常值过滤，默认 True
        - outlier_criteria: 异常值过滤使用的变换类型，'homography' 或 'affine'，默认 'homography'
        - outlier_threshold: 异常值过滤的误差阈值，默认 20.0 像素
        
        Returns:
        - goodMatch: 过滤后的匹配点列表 (cv2.DMatch 对象列表)
        - cv_kpts_query: query 图像的关键点列表
        - cv_kpts_refer: refer 图像的关键点列表
        - query_image: query 图像
        - refer_image: refer 图像
        """
        # 获取图像和关键点
        query_image, refer_image = self.image_read(query_path, refer_path, query_is_image)
        query_tensor = self.trasformer(Image.fromarray(query_image))
        refer_tensor = self.trasformer(Image.fromarray(refer_image))
        
        keypoints, descriptors = self.model_run_pair(query_tensor, refer_tensor)
        
        query_keypoints, refer_keypoints = keypoints[0], keypoints[1]
        query_desc, refer_desc = descriptors[0].permute(1, 0).numpy(), descriptors[1].permute(1, 0).numpy()
        
        # 映射关键点到原始图像尺寸
        cv_kpts_query_full = [cv2.KeyPoint(int(i[0] / self.model_image_width * self.image_width),
                                          int(i[1] / self.model_image_height * self.image_height), 30)
                             for i in query_keypoints]
        cv_kpts_refer_full = [cv2.KeyPoint(int(i[0] / self.model_image_width * self.image_width),
                                          int(i[1] / self.model_image_height * self.image_height), 30)
                             for i in refer_keypoints]
        
        # 第一步：基础匹配
        goodMatch = []
        try:
            matches = self.knn_matcher.knnMatch(query_desc, refer_desc, k=2)
            for m, n in matches:
                if m.distance < self.knn_thresh * n.distance:
                    goodMatch.append(m)
        except Exception:
            pass
        
        if len(goodMatch) == 0:
            return [], cv_kpts_query_full, cv_kpts_refer_full, query_image, refer_image
        
        # 第二步：逆一致性检查
        if use_inverse_consistency:
            filtered_matches, _, _ = self.check_inverse_consistency(
                cv_kpts_query_full, cv_kpts_refer_full, query_desc, refer_desc, iccl=iccl
            )
            
            if len(filtered_matches) > 0:
                goodMatch = filtered_matches
        
        # 第三步：异常值过滤
        if use_outlier_filter and len(goodMatch) >= 4:
            query_pts = [cv_kpts_query_full[m.queryIdx].pt for m in goodMatch]
            refer_pts = [cv_kpts_refer_full[m.trainIdx].pt for m in goodMatch]
            
            filtered_query_pts, filtered_refer_pts, _ = self.filter_outliers(
                query_pts, refer_pts, 
                criteria=outlier_criteria, 
                threshold=outlier_threshold
            )
            
            # 更新匹配列表，只保留过滤后的点
            if len(filtered_query_pts) < len(query_pts):
                # 创建过滤后的点集合（使用坐标匹配）
                filtered_query_pts_dict = {(pt[0], pt[1]): i for i, pt in enumerate(filtered_query_pts)}
                filtered_refer_pts_dict = {(pt[0], pt[1]): i for i, pt in enumerate(filtered_refer_pts)}
                
                new_goodMatch = []
                new_cv_kpts_query = []
                new_cv_kpts_refer = []
                query_idx_map = {}  # 原始索引 -> 新索引
                refer_idx_map = {}  # 原始索引 -> 新索引
                
                for match in goodMatch:
                    q_pt = cv_kpts_query_full[match.queryIdx].pt
                    r_pt = cv_kpts_refer_full[match.trainIdx].pt
                    q_key = (q_pt[0], q_pt[1])
                    r_key = (r_pt[0], r_pt[1])
                    
                    if q_key in filtered_query_pts_dict and r_key in filtered_refer_pts_dict:
                        # 获取新的索引
                        if match.queryIdx not in query_idx_map:
                            new_query_idx = len(new_cv_kpts_query)
                            query_idx_map[match.queryIdx] = new_query_idx
                            new_cv_kpts_query.append(cv_kpts_query_full[match.queryIdx])
                        else:
                            new_query_idx = query_idx_map[match.queryIdx]
                        
                        if match.trainIdx not in refer_idx_map:
                            new_refer_idx = len(new_cv_kpts_refer)
                            refer_idx_map[match.trainIdx] = new_refer_idx
                            new_cv_kpts_refer.append(cv_kpts_refer_full[match.trainIdx])
                        else:
                            new_refer_idx = refer_idx_map[match.trainIdx]
                        
                        new_match = cv2.DMatch(new_query_idx, new_refer_idx, match.distance)
                        new_goodMatch.append(new_match)
                
                goodMatch = new_goodMatch
                cv_kpts_query_full = new_cv_kpts_query
                cv_kpts_refer_full = new_cv_kpts_refer
        
        return goodMatch, cv_kpts_query_full, cv_kpts_refer_full, query_image, refer_image

if __name__ == '__main__':
    import yaml

    config_path = 'config/test.yaml'
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError("Config File doesn't Exist")

    P = Predictor(config)
    f1 = './data/samples/query.jpg'
    f2 = './data/samples/refer.jpg'
    P.match(f1, f2, show=True)
    merged = P.align_image_pair(f1, f2)
    plt.imshow(merged)
    plt.show()
