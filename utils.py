import cv2 as cv
def render_obs(obs, use_rnn):
    cv.namedWindow('env0', cv.WINDOW_NORMAL)
    obs = obs[0]
    for i in range(obs.shape[0]):
        frame = obs[i].numpy() if use_rnn else obs[i][:,:,3].numpy()
        cv.imshow('env0', frame)
        cv.waitKey(delay=10)

class AtariDecay:
    def __init__(self, initial_value, end_value, decay_steps, decay_after_steps=1):
        assert end_value < initial_value
        self.initial_value = initial_value
        self.end_value = end_value
        self.decay_steps = decay_steps
        self.decay_after_steps = decay_after_steps

        self.current_step = 0

        value_sequence = [initial_value]
        decay_value = (initial_value - end_value) / decay_steps
        for _ in range(1, decay_steps):
            value_sequence.append(value_sequence[-1] - decay_value)
        value_sequence.append(end_value)
        self.value_sequence = value_sequence
        self.value_index = -1
    
    def __call__(self):
        if self.current_step % self.decay_after_steps == 0:
            self.value_index += 1
        self.current_step += 1
        if self.value_index >= len(self.value_sequence):
            return self.end_value
        return self.value_sequence[self.value_index]