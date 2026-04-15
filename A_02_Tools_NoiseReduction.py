# 姿态分类结果平滑（指数移动平均（EMA）方法）
class EMADictSmoothing(object):
    """平滑姿势分类的类。"""

    def __init__(self, window_size=10, alpha=0.2):
        """
        初始化平滑对象
        :param window_size: 时间窗口大小
        :param alpha: 指数移动平均的权重
        """
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window = []

    def __call__(self, data):
        """
        平滑给定的姿势分类。
        :param data: 姿势分类的字典
        :return: 平滑后的字典
        """
        self._data_in_window.insert(0, data)
        if len(self._data_in_window) > self._window_size:
            self._data_in_window.pop()

        keys = set(key for data in self._data_in_window for key in data)

        smoothed_data = {}
        for key in keys:
            factor = 1.0
            top_sum = 0.0
            bottom_sum = 0.0
            for data in self._data_in_window:
                value = data.get(key, 0.0)
                top_sum += factor * value
                bottom_sum += factor
                factor *= (1.0 - self._alpha)

            smoothed_data[key] = top_sum / bottom_sum

        return smoothed_data
