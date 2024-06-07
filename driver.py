import pygame
import numpy as np

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1200, 600))
clock = pygame.time.Clock()
running = True
t = 0
dt = 0

segment_length = 50

loc = np.array([[100, 300],
                [150, 300],
                [200, 300],
                [250, 300]]).astype(np.float64)

vel = np.array([[0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0]])


def draw_worm():
    pygame.draw.rect(screen, "cyan", pygame.Rect(0, 0, 1200, 300))
    pygame.draw.rect(screen, "green", pygame.Rect(0, 200, 1200, 300))
    pygame.draw.rect(screen, "brown", pygame.Rect(0, 210, 1200, 400))

    for i in range(len(loc) - 1):
        x1, y1 = loc[i]
        x2, y2 = loc[i + 1]

        pygame.draw.line(screen, "pink", (x1, y1), (x2, y2), 20)
        pygame.draw.circle(screen, "pink", (loc[i]), 10)

    pygame.draw.circle(screen, "pink", loc[-1], 20)
    pygame.draw.circle(screen, "black", loc[-1] + np.array([7, 0]), 4)
    pygame.draw.circle(screen, "black", loc[-1] + np.array([-7, 0]), 4)



def impulse(i, tau, dt):
    x, y = loc[i]

    I1 = 0

    for j in range(i):
        r = (loc[j] - loc[i]) / segment_length
        mag = np.linalg.norm(r)

        cp = (loc[j] + loc[j + 1]) / 2 - loc[i]
        cp = cp / np.linalg.norm(cp)

        seg = loc[j] - loc[j + 1]
        seg = seg / np.linalg.norm(seg)

        coeff = np.dot(cp, seg)

        I1 += coeff * mag**2

    alpha1 = -tau / I1

    for j in range(i):
        r = (loc[j] - loc[i]) / segment_length
        rnorm = np.zeros(3)
        rnorm[:2] = r / np.linalg.norm(r)

        dv = np.cross([0, 0, alpha1], rnorm) * dt

        vel[j] += dv[:2]

    I1 = 0

    for j in range(i+1, len(loc)):
        r = (loc[j] - loc[i]) / segment_length
        mag = np.linalg.norm(r)

        cp = (loc[j] + loc[j - 1]) / 2 - loc[i]
        cp = cp / np.linalg.norm(cp)

        seg = loc[j] - loc[j - 1]
        seg = seg / np.linalg.norm(seg)

        coeff = np.dot(cp, seg)

        I1 += coeff * mag**2

    alpha1 = tau / I1

    for j in range(i+1, len(loc)):
        r = (loc[j] - loc[i]) / segment_length
        rnorm = np.zeros(3)
        rnorm[:2] = r / np.linalg.norm(r)

        dv = np.cross([0, 0, alpha1], rnorm) * dt

        vel[j] += dv[:2]


while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    dt += 0.1

    t += dt

    #impulse(2, 20 * np.sin(t), dt)
    #impulse(1, -20 * np.sin(t), dt)

    #loc[:,:] += vel[:,:] * dt

    loc[:, 0] += np.sin(np.array([0,1,2,3])*2 + t) / 2 + 1
    loc[:, 1] += np.sin(np.array([0,1,2,3])*2 + t)



    draw_worm()

    keys = pygame.key.get_pressed()

    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()
