import numpy as np

N = 302
T = 20
tau = 0.1
dt = 1e-3
p = 1
g = 1.5
nt = round(T / dt)
omega = np.random.normal(loc=0, scale=g / np.sqrt(N), size=(N, N))
bias = np.random.randn(N, 1) * 0
lamda = 1
Pinv = np.eye(N) / lamda

# supervisor(s)
sig = np.zeros((4, nt))

# Define the time array
time = np.arange(1, nt + 1) * dt

# Set each row of sig to a sine wave with different frequencies
freq = 0.2
offsets = [0, np.pi / 2, np.pi, 3 * np.pi / 2]  # Example frequencies for each row
for j, offset in enumerate(offsets):
    sig[j] = np.sin(2 * np.pi * freq * time + offset)

k = 4
eta = 2 * np.random.rand(N, k) - 1
phi = np.zeros((N, k))
y = np.random.randn(N)
r = np.tanh(y)
xhat = np.dot(phi.T, r)
imin = 2 / dt
imax = 0.5 * T / dt
step = 2

store_r = np.zeros((nt, 10))
store_x = np.zeros((nt, k))
store_phi = np.zeros((nt, 10))
nz = 0

# perturbation (doesn't currently work/do anything to the network even for high values...bug?)
u = np.zeros(N)
indices = slice(1, 81, 2)
value = 1000
u[indices] = value


def tanhnet(y, omega, eta, sig, i, tau):
    sig_i = sig[:, i].T
    a = np.dot(omega, np.tanh(y))
    b = np.dot(eta, sig_i)
    t = np.tanh(y)
    dy = (-y + np.dot(omega, np.tanh(y)) + np.dot(eta, sig_i)) / tau
    return dy


step1 = 10
for i in range(1, nt):
    dy = tanhnet(y, omega, eta, sig, i, tau)
    y = y + dy * dt
    # if i % step1 == 0:
    # y=y+u
    # if i == 100:
    #     print(u)
    r = np.tanh(y)
    xhat = np.dot(phi.T, r)
    if i > imin:
        if i % step == 0:
            # s=sig[i]
            # print(s.shape)
            # e = xhat - sig[:,i]
            # print(r.shape)
            # q = np.dot(Pinv, r)
            # Pinv = Pinv - np.outer(q, q.T) / (1 + np.dot(r.T, q))
            # phi = phi - np.dot(Pinv, np.dot(r, e.T))
            # if imin < i < imax and i % step == 0:
            e = xhat - sig[:, i]
            q = Pinv @ r
            Pinv = Pinv - np.outer(q, q) / (1 + np.dot(r, q))
            phi = phi - np.outer(Pinv @ r, e)
    store_x[i, :] = xhat  # Store the reshaped xhat into the store_x array

    # print(r.shape)
    # print(xhat.shape)
    # store_r[i, :] = r[:10]
    # store_x[i, ,:] = xhat

    # store_phi[i, :] = phi[:10]
# print(sig.shape)
# print(store_x.shape)
x1 = store_x[:, 0]
x2 = store_x[:, 1]
x3 = store_x[:, 2]
x4 = store_x[:, 3]
# print(x1.shape)

import matplotlib.pyplot as plt

# Create subplots
fig, axes = plt.subplots(4, 1)  # 2 rows, 1 column

# Plot data on each subplot
axes[0].plot(time, sig[0, :], 'r-', label='sig')  # Plot sig as a solid red line
axes[0].plot(time, x1, 'k--', label='store_x column')  # Plot the column of store_x as a black dotted line
axes[0].set_title('Head Node')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Position')

axes[1].plot(time, sig[1, :], 'r-', label='sig')  # Plot sig as a solid red line
axes[1].plot(time, x2, 'k--', label='store_x column')  # Plot the column of store_x as a black dotted line
axes[1].set_title('Chest Node')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Position')

axes[2].plot(time, sig[2, :], 'r-', label='sig')  # Plot sig as a solid red line
axes[2].plot(time, x3, 'k--', label='store_x column')  # Plot the column of store_x as a black dotted line
axes[2].set_title('Butt Node')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Position')

axes[3].plot(time, sig[3, :], 'r-', label='sig')  # Plot sig as a solid red line
axes[3].plot(time, x4, 'k--', label='store_x column')  # Plot the column of store_x as a black dotted line
axes[3].set_title('Tail Node')
axes[3].set_xlabel('Time')
axes[3].set_ylabel('Position')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# animation

ys = np.array([x1, x2, x3, x4]).T[1500::10]


def y():
    for yy in ys:
        yield yy


yss = y()

import pygame

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

pygame.draw.rect(screen, "cyan", pygame.Rect(0, 0, 1200, 300))
pygame.draw.rect(screen, "green", pygame.Rect(0, 300, 1200, 300))


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

    # impulse(2, 20 * np.sin(t), dt)
    # impulse(1, -20 * np.sin(t), dt)

    # loc[:,:] += vel[:,:] * dt

    loc[:, 0] += 1
    loc[:, 1] = next(yss) * 30 + 300
    print(next(yss) * 30 + 300)

    draw_worm()

    keys = pygame.key.get_pressed()

    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()


