import cv2 as cv
def render_obs(obs):
    cv.namedWindow('env0', cv.WINDOW_NORMAL)
    obs = obs[0]
    for i in range(obs.shape[0]):
        frame = obs[i][:,:,3].numpy()
        #frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow('env0', frame)
        cv.waitKey(delay=100)
