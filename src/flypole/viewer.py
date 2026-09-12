"""Bounded live rendering: actual CartPole state and a sampled activity graph."""

import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame


class Viewer:
    def __init__(self, brain, fps=50, stride=1, offscreen=False):
        pygame.font.init()
        self.offscreen = offscreen
        self.surface = pygame.Surface((1200, 650)) if offscreen else pygame.display.set_mode((1200, 650))
        if not offscreen:
            pygame.display.set_caption("FlyPole | fixed brain, learned readout")
        self.font = pygame.font.Font(None, 24)
        self.small = pygame.font.Font(None, 19)
        self.clock = pygame.time.Clock()
        self.fps, self.stride = fps, stride
        self.brain = brain
        self.frames = 0
        self.closed = False
        self.nodes = np.array([], dtype=int)
        self.edges = []
        self.positions = {}
        if brain.graph is not None:
            n = len(brain.activity)
            preferred = np.unique(np.r_[brain.groups.ravel(), brain.features])
            others = np.setdiff1d(np.arange(n), preferred)
            self.nodes = np.r_[preferred, np.random.default_rng(0).choice(others, min(len(others), 240-len(preferred)), replace=False)]
            # Stable abstract layout, deliberately independent of anatomy.
            for i, node in enumerate(self.nodes):
                angle = 2*np.pi*i/len(self.nodes)
                radius = 120 + 50*(i % 3)/2
                self.positions[int(node)] = (int(910+radius*np.cos(angle)), int(350+radius*np.sin(angle)))
            coo = brain.graph.adjacency.tocoo()
            mask = np.isin(coo.row, self.nodes) & np.isin(coo.col, self.nodes)
            pairs = np.flatnonzero(mask)
            chosen = np.random.default_rng(0).choice(pairs, min(len(pairs), 350), replace=False)
            self.edges = [(int(coo.col[i]), int(coo.row[i])) for i in chosen]

    def text(self, message, position, small=False, color=(206, 215, 226)):
        self.surface.blit((self.small if small else self.font).render(str(message), True, color), position)

    def draw(self, observation, activity, reward, step, status, action):
        if not self.offscreen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.close()
                    raise RuntimeError("viewer closed; existing checkpoints remain saved")
        if step % self.stride:
            return
        self.frames += 1
        self.surface.fill((16, 22, 33))
        pygame.draw.line(self.surface, (58, 70, 86), (600, 85), (600, 600))
        self.text("FlyPole — fixed neural connectivity, trained linear readout", (25, 20))
        self.text(status, (25, 52))
        self.text("CartPole-v1", (25, 100))
        self.text(f"reward {reward:.0f}/500    action {'RIGHT' if action else 'LEFT'}", (25, 135))
        x = int(300 + float(observation[0])*95)
        angle = float(observation[2])
        pygame.draw.line(self.surface, (120, 134, 154), (30, 450), (570, 450), 2)
        pygame.draw.rect(self.surface, (90, 163, 218), (x-32, 411, 64, 32), border_radius=4)
        pygame.draw.circle(self.surface, (211, 222, 238), (x-20, 447), 6)
        pygame.draw.circle(self.surface, (211, 222, 238), (x+20, 447), 6)
        endpoint = (int(x+150*np.sin(angle)), int(412-150*np.cos(angle)))
        pygame.draw.line(self.surface, (255, 178, 75), (x, 412), endpoint, 9)
        for i, name in enumerate(("position", "velocity", "angle", "angular velocity")):
            self.text(f"{name}: {observation[i]:+.3f}", (30, 485+22*i), small=True)
        if self.brain.graph is None:
            self.text("Direct observation baseline", (720, 290))
            self.text("No neuron activity in this control", (720, 322), small=True)
        else:
            dataset = self.brain.graph.manifest.get("dataset", "unknown")
            variant = self.brain.graph.manifest.get("control_variant", "original")
            self.text(f"{dataset} | {variant}", (625, 100))
            active = int(np.count_nonzero(activity > 1e-4))
            self.text(f"active {active}/{len(activity)} | threshold 0.0001", (625, 130), small=True)
            maximum = max(float(np.max(activity)), 1e-6)
            for source, target in self.edges:
                level = np.sqrt(min(activity[source], activity[target]) / maximum)
                color = (45, int(60+100*level), int(75+100*level))
                pygame.draw.line(self.surface, color, self.positions[source], self.positions[target])
            inputs = set(self.brain.groups.ravel())
            features = set(self.brain.features)
            for node in self.nodes:
                level = float(np.clip(activity[node]/maximum, 0, 1))
                color = (int(45+210*level), int(80+150*level), int(120-60*level))
                pygame.draw.circle(self.surface, color, self.positions[int(node)], 3+int(level*3))
                if node in inputs or node in features:
                    pygame.draw.circle(self.surface, (70, 205, 235) if node in inputs else (215, 140, 240), self.positions[int(node)], 7, 1)
            self.text(f"Sample: {len(self.nodes)} neurons, {len(self.edges)} edges", (625, 560), small=True)
            self.text("Cyan rings: sensory input | purple: downstream readout", (625, 583), small=True)
        self.text("Engineered encoding, dynamics, readout and RL. Abstract layout; no anatomy implied.", (25, 620), small=True)
        if not self.offscreen:
            pygame.display.flip()
            if self.fps:
                self.clock.tick(self.fps)

    def close(self):
        if not self.offscreen:
            pygame.display.quit()
        self.closed = True
