import os
import argparse
import math
import cv2 as cv
import numpy as np
import random
import heapq
import uuid
import time
import json

from tqdm import tqdm

# random.seed(666)
# np.random.seed(666)


def check_path_by_astar(grid_start, grid_end, grid_map):
    """
    基于 A* 算法寻找从起点到终点的路径
    :param grid_start: 起点坐标 (行, 列)
    :param grid_end: 终点坐标 (行, 列)
    :param grid_map: numpy二维数组 (0表示可通行，1表示障碍物)
    :return: 路径列表 [(r1, c1), (r2, c2), ...]，如果找不到路径则返回 None
    """
    # 使用 numpy 的属性获取行数和列数
    rows, cols = grid_map.shape

    # 辅助函数：计算曼哈顿距离作为启发式估值 (h值)
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # 优先队列（最小堆），存储 (f值, g值, 当前坐标)
    open_set = []
    heapq.heappush(open_set, (0, 0, grid_start))

    # 记录已经访问过的节点（当前节点是从哪个节点走过来的）
    came_from = {}

    # 记录从起点到当前节点的实际代价 (g值)
    g_score = {grid_start: 0}

    # 记录从起点经过当前节点到终点的预估总代价 (f值)
    f_score = {grid_start: heuristic(grid_start, grid_end)}

    # 上下左右四个方向的偏移量
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_set:
        # 弹出 f 值最小的节点
        _, current_g, current = heapq.heappop(open_set)

        # 如果到达终点，开始回溯并构建路径
        if current == grid_end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(grid_start)
            return path[::-1]  # 反转路径，使其从起点到终点

        # 遍历当前节点的四个邻居
        for dr, dc in directions:
            r, c = current[0] + dr, current[1] + dc
            neighbor = (r, c)

            # 检查边界和障碍物 (使用 numpy 数组进行索引)
            if 0 <= r < rows and 0 <= c < cols and grid_map[r, c] == 0:
                # 假设每走一步的代价为 1
                tentative_g_score = g_score[current] + 1

                # 如果找到了一条到达邻居的更短路径，或者邻居还没被探索过
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(
                        neighbor, grid_end
                    )

                    # 将邻居加入优先队列
                    heapq.heappush(
                        open_set, (f_score[neighbor], tentative_g_score, neighbor)
                    )

    # 如果队列为空还没找到终点，说明没有路径
    return []


def paint_cylinder_obstacle(radius_grid, plane_map, margin_grid=1):
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (margin_grid * 2 + 1, margin_grid * 2 + 1)
    )

    dilate_image = cv.dilate(plane_map, kernel, iterations=1)

    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (radius_grid * 2 + 1, radius_grid * 2 + 1)
    )
    dilate_image = cv.dilate(dilate_image, kernel, iterations=1)
    x_idx, y_idx = np.nonzero(np.bitwise_not(dilate_image))
    if len(x_idx) == 0:
        return False, (-1, -1)
    c_id = random.choice(range(len(x_idx)))
    x_id, y_id = x_idx[c_id], y_idx[c_id]
    cv.circle(plane_map, (y_id, x_id), radius_grid, 255, -1)

    return True, (x_id, y_id)


def paint_cube_obstacle(width_grid, height_grid, plane_map, margin_grid=1):

    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (margin_grid * 2 + 1, margin_grid * 2 + 1)
    )
    dilate_image = cv.dilate(plane_map, kernel, iterations=1)

    half_width_grid = math.ceil(width_grid / 2)
    half_height_grid = math.ceil(height_grid / 2)

    kernel = cv.getStructuringElement(
        cv.MORPH_RECT, (half_height_grid * 2 + 1, half_width_grid * 2 + 1)
    )
    dilate_image = cv.dilate(dilate_image, kernel, iterations=1)

    x_idx, y_idx = np.nonzero(dilate_image == 0)
    if len(x_idx) == 0:
        return False, (-1, -1)
    c_id = random.choice(range(len(x_idx)))
    x_id, y_id = x_idx[c_id], y_idx[c_id]
    cv.rectangle(
        plane_map,
        (y_id - half_height_grid, x_id - half_width_grid),
        (y_id + half_height_grid, x_id + half_width_grid),
        255,
        -1,
    )
    return True, (x_id, y_id)


def generate_obstacle_map(
    area_size=(6.0, 6.0),
    num_obstacles=20,
    robot_radius=0.2,
    resolution=0.02,
    cylinder_candidates=None,
    cube_candidates=None,
    padding_grid=5,
    map_padding=1.0,
    margin_grid=2,
):
    assert num_obstacles < len(cylinder_candidates) + len(cube_candidates)
    x_grid, y_grid = math.ceil(area_size[0] / resolution), math.ceil(
        area_size[1] / resolution
    )
    robot_radius_grid = math.ceil(robot_radius / resolution)
    cylinder_idxs = set(range(len(cylinder_candidates)))
    cube_idxs = set(range(len(cube_candidates)))
    plane_map = np.zeros((x_grid, y_grid), dtype=np.uint8)
    plane_map[:, [*range(padding_grid), *range(-1, -1 - padding_grid, -1)]] = 255
    plane_map[[*range(padding_grid), *range(-1, -1 - padding_grid, -1)], :] = 255

    valid_count = 0
    success = False
    obstacle_info = []
    for _ in range(num_obstacles * 10):
        is_cylinder = random.randint(0, 1)
        if is_cylinder and len(cylinder_idxs) ==0 and len(cube_idxs) >0:
            is_cylinder = False
        if not is_cylinder and len(cube_idxs) ==0 and len(cylinder_idxs) >0:
            is_cylinder = True
        is_valid = False
        if is_cylinder and len(cylinder_idxs) > 0:
            idx = random.choice(list(cylinder_idxs))
            radius = cylinder_candidates[idx]
            radius_grid = math.ceil(radius / resolution)
            is_valid, select_grid_idx = paint_cylinder_obstacle(
                radius_grid, plane_map, margin_grid=margin_grid
            )
            type_str = "cylinder"
            geo_property = round(radius, 4)
            if is_valid:
                cylinder_idxs.remove(idx)

        elif not is_cylinder and len(cube_idxs) > 0:
            idx = random.choice(list(cube_idxs))
            
            width, height = cube_candidates[idx]
            width_grid, height_grid = math.ceil(width / resolution), math.ceil(
                height / resolution
            )
            is_valid, select_grid_idx = paint_cube_obstacle(
                width_grid, height_grid, plane_map, margin_grid=margin_grid
            )
            type_str = "cube"
            geo_property = (round(width, 4), round(height, 4))
            if is_valid:
                cube_idxs.remove(idx)
        else:
            raise ValueError("没有更多的障碍物了")
        if is_valid:
            cg_x, cg_y = select_grid_idx
            c_x, c_y = (cg_x + 0.5) * resolution - area_size[0] * 0.5, (
                cg_y + 0.5
            ) * resolution - area_size[1] * 0.5
            obstacle_info.append(
                {
                    "type": type_str,
                    "pos": (round(c_x, 4), round(c_y, 4)),
                    "geo_property": geo_property,
                    "idx": int(idx),
                }
            )
            valid_count += 1
        if valid_count >= num_obstacles:
            plane_inner = plane_map[:, padding_grid:-padding_grid]
            plane_inner[: padding_grid + 1, :] = 0
            plane_inner[-padding_grid - 1 :, :] = 0
            map_padding_grid = math.ceil(map_padding / resolution)
            plane_map = np.pad(
                plane_map,
                ((map_padding_grid, map_padding_grid), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            plane_map[:, [*range(padding_grid), *range(-1, -1 - padding_grid, -1)]] = (
                255
            )
            plane_map[[*range(padding_grid), *range(-1, -1 - padding_grid, -1)], :] = (
                255
            )

            kernel = cv.getStructuringElement(
                cv.MORPH_ELLIPSE, (robot_radius_grid * 2 + 1, robot_radius_grid * 2 + 1)
            )
            dilate_map = cv.dilate(plane_map, kernel, iterations=1)

            if margin_grid >= robot_radius_grid * 2:
                paths = [(1,1)]
            else:
                paths = check_path_by_astar(
                    (robot_radius_grid * 2 + 2, int(plane_map.shape[1] / 2)),
                    (
                        plane_map.shape[0] - (robot_radius_grid * 2 + 3),
                        int(plane_map.shape[1] / 2),
                    ),
                    dilate_map,
                )

            success = len(paths) > 0
            # for (path_x, path_y) in paths:
            #     plane_map[path_x, path_y] = 255
            break
    return success, obstacle_info, plane_map


def main(args):
    cylinder_candidates = np.arange(0.04, 0.641, 0.02)
    cube_candidates_width = np.arange(0.04, 0.541, 0.02)
    cube_candidates_height = np.arange(0.04, 0.541, 0.02)
    cube_candidates = np.stack([cube_candidates_width, cube_candidates_height], axis=1)
    info_to_save = {}
    info_to_save["cylinder_candidates"] = [
        round(item, 4) for item in cylinder_candidates.tolist()
    ]
    info_to_save["cube_candidates"] = [
        [round(item[0], 4), round(item[1], 4)] for item in cube_candidates.tolist()
    ]
    info_to_save["area_size"] = [
        round(args.area_size[0] + 2.0 * args.padding_x, 4),
        round(args.area_size[1], 4),
    ]
    info_to_save['robot_radius'] = round(args.robot_radius, 4)
    info_to_save["num_obstacles"] = args.num_obstacles
    info_to_save["obstacle_maps"] = []
    info_to_save['resolution'] = args.resolution
    margin_grid = args.margin_grid
    if args.sparse:
        margin_grid = math.ceil(args.robot_radius * 2 / args.resolution)
        

    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    package_dir = os.path.join(args.output_dir, time_str)

    if not os.path.exists(package_dir):
        os.makedirs(package_dir)
    if args.save_images:
        images_dir = os.path.join(package_dir, "visualizes")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

    for _ in tqdm(range(args.num)):
        success, obstacle_info, plane_map = generate_obstacle_map(
            cylinder_candidates=cylinder_candidates,
            cube_candidates=cube_candidates,
            num_obstacles=args.num_obstacles,
            resolution=args.resolution,
            area_size=args.area_size,
            map_padding=args.padding_x,
            robot_radius=args.robot_radius,
            margin_grid=margin_grid
        )
        if not success:
            continue
        map_path = f"visualizes/{uuid.uuid4().hex}.png"
        info_to_save["obstacle_maps"].append(
            {"map": map_path, "obstacles": obstacle_info}
        )
        if args.save_images:
            cv.imwrite(os.path.join(package_dir, map_path), plane_map)
    info_to_save["num"] = args.num
    with open(os.path.join(package_dir, "parameters.json"), "w") as f:
        json.dump(info_to_save, f)
    print(f"保存了 {len(info_to_save['obstacle_maps'])} 个地图")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Obstacle Map")
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--num_obstacles", type=int, default=10)
    parser.add_argument("--resolution", type=float, default=0.02)
    parser.add_argument("--robot_radius", type=float, default=0.2)
    parser.add_argument("--margin_grid", type=int, default=2)
    parser.add_argument("--area_size", nargs="+", type=float, default=[6.0, 6.0])
    parser.add_argument("--padding_x", type=int, default=1.0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_images", default=False, action="store_true")
    parser.add_argument("--sparse", default=False, action="store_true")
    main(parser.parse_args())
