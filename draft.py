# Rotate around the x-axis
# trans_init = np.asarray(
#     [
#         [1, 0, 0],
#         [0, 1 / 2, -math.sqrt(3) / 2],
#         [0.0, math.sqrt(3) / 2, 1 / 2],
#     ]
# )
# Rotate around the y-axis
# trans_init = np.asarray(
#     [
#         [2 / math.sqrt(5), 0, 1 / math.sqrt(5)],
#         [0.0, 1, 0.0],
#         [-1 / math.sqrt(5), 0.0, 2 / math.sqrt(5)],
#     ]
# )
# Rotate around the z-axis
# trans_init = np.asarray(
#     [
#         [2 / math.sqrt(5), -1 / math.sqrt(5), 0],
#         [1 / math.sqrt(5), 2 / math.sqrt(5), 0.0],
#         [0.0, 0.0, 1.0],
#     ]
# )

# 26度回転
trans_init = np.asarray(
    [
        [2 / math.sqrt(5), -1 / math.sqrt(5), 0.0, -0.0],
        [1 / math.sqrt(5), 2 / math.sqrt(5), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# -26度回転
trans_init = np.asarray(
    [
        [2 / math.sqrt(5), 1 / math.sqrt(5), 0.0, -0.0],
        [-1 / math.sqrt(5), 2 / math.sqrt(5), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# 64度回転
trans_init = np.asarray(
    [
        [1 / math.sqrt(5), -2 / math.sqrt(5), 0.0, -0.0],
        [2 / math.sqrt(5), 1 / math.sqrt(5), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# -54度回転
trans_init = np.asarray(
    [
        [0.58, 0.80, 0.0, -0.0],
        [-0.80, 0.58, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
