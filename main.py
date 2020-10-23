import pygame
import random
import neat
import os

pygame.init()
bgImage = pygame.transform.scale2x(pygame.image.load("base.png"))
pipeImage = pygame.transform.scale2x(pygame.image.load("pipe.png"))
birdImages = [pygame.transform.scale2x(pygame.image.load("bird1.png")),
              pygame.transform.scale2x(pygame.image.load("bird2.png")),
              pygame.transform.scale2x(pygame.image.load("bird3.png"))]
winWidth = 600
winHeight = 1000
gameWin = pygame.display.set_mode((winWidth, winHeight))
clock = pygame.time.Clock()
gen = 0

class Background:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.image = pygame.transform.scale2x(pygame.image.load("bg.png"))
        self.x2 = self.image.get_width()

    def draw(self, gameWin):
        gameWin.blit(self.image, (self.x, self.y))
        gameWin.blit(self.image, (self.x2, self.y))


class Base:
    vel = 10
    def __init__(self, y):
        self.y = y
        self.image = bgImage
        self.width = self.image.get_width()
        self.x1 = 0
        self.x2 = self.width
        self.x3 = 2 * self.width

    def draw(self, gameWin):
        gameWin.blit(self.image, (self.x1, self.y))
        gameWin.blit(self.image, (self.x2, self.y))
        gameWin.blit(self.image, (self.x3, self.y))

    def move(self):
        self.x1 -= 10
        self.x2 -= 10
        self.x3 -= 10
        if self.x1 + self.width <= 0:
            self.x1 = self.x3 + self.width
        if self.x2 + self.width <= 0:
            self.x2 = self.x1 + self.width
        if self.x3 + self.width <= 0:
            self.x3 = self.x2 + self.width
        self.draw(gameWin)


class Pipe:
    vel = 10
    gap = 200
    def __init__(self, x):
        self.x = x
        self.top = 0
        self.bottom = 0
        self.height = 0
        self.topPipe = pygame.transform.flip(pipeImage, False, True)
        self.bottomPipe = pipeImage
        self.width = pipeImage.get_width()
        self.passed = False
        self.setHeight()

    def draw(self, gameWin):
        gameWin.blit(self.bottomPipe, (self.x, self.bottom))
        gameWin.blit(self.topPipe, (self.x, self.top))

    def setHeight(self):
        self.height = random.randint(55, 550)
        self.top = self.height - self.topPipe.get_height()
        self.bottom = self.height + 200

    def move(self):
        self.x -= self.vel
        if self.x + self.width < 0:
            self.x = winWidth
            self.passed = False
            self.setHeight()
        self.draw(gameWin)

    def collide(self, bird):
        birdMask = bird.getMask()
        topMask = pygame.mask.from_surface(self.topPipe)
        bottomMask = pygame.mask.from_surface(self.bottomPipe)
        bOffset = (self.x - bird.x, self.bottom - round(bird.y))
        tOffset = (self.x - bird.x, self.top - round(bird.y))
        bPoint = birdMask.overlap(bottomMask, bOffset)
        tPoint = birdMask.overlap(topMask, tOffset)
        if tPoint or bPoint:
            return True
        return False



class Bird:
    birdAnimation = 4
    vel = -10.5
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.imgCount = 0
        self.image = birdImages[0]
        self.tilt = 0
        self.height = self.y
        self.tickCount = 0

    def draw(self, gameWin):
        self.imgCount += 1
        if self.imgCount < self.birdAnimation:
            self.image = birdImages[0]
        elif self.imgCount < self.birdAnimation*2:
            self.image = birdImages[1]
        elif self.imgCount < self.birdAnimation*3:
            self.image = birdImages[2]
        elif self.imgCount < self.birdAnimation*4:
            self.image = birdImages[1]
        elif self.imgCount == self.birdAnimation*4 + 1:
            self.image = birdImages[0]
            self.imgCount = 0
        gameWin.blit(self.image, (self.x, self.y))

    def jump(self):
        self.vel = -10.5
        self.tickCount = 0
        self.height = self.y

    def move(self):
        self.tickCount += 1
        d = -10.5*self.tickCount + 1.5*(self.tickCount**2)
        if d >= 10:
            d = 10
        if d < 0:
            d = -7
        self.y += d
        self.draw(gameWin)

    def getMask(self):
        return pygame.mask.from_surface(self.image)

def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(main, 100)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

def main(genomes, config):
    genn = 0
    ge = []
    nets = []
    birds = []
    for genomeId, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(100, 500))
        g.fitness = 0
        ge.append(g)
    pipe = Pipe(700)
    bg = Background(0, 0)
    base = Base(800)
    score = 0
    while True and len(birds) > 0:
        bg.draw(gameWin)
        addScore = False

        font1 = pygame.font.Font("comicbd.ttf", 20)
        scoreText = font1.render("Score: " + str(score), True, (123, 123, 123))
        alive = font1.render("Alive: " + str(len(birds)), True, (12, 12, 12))
        gen = font1.render("Generation: " + str(genn), True, (12, 12, 12))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        pipe.move()

        for b in birds:
            x = birds.index(b)
            if pipe.collide(b):
                ge[x].fitness -= 0.01
                birds.pop(x)
                ge.pop(x)
                nets.pop(x)
            else:
                ge[x].fitness += 0.01
                output = nets[x].activate((pipe.x, b.y, abs(b.y - pipe.height), abs(b.y - pipe.bottom)))
                if output[0] > 0.5:
                    b.tickCount = 0

        for b in birds:
            x = birds.index(b)
            if b.y > 760 or b.y < -50:
                ge[x].fitness -= 0.04
                birds.pop(x)
                ge.pop(x)
                nets.pop(x)
            if not pipe.passed and pipe.x + pipe.width < b.x:
                pipe.passed = True
                addScore = True
                ge[x].fitness += 0.5

        if addScore:
            score += 10
        for b in birds:
            b.move()
        base.move()

        gameWin.blit(scoreText, (10, 10))
        gameWin.blit(alive, (500, 10))
        gameWin.blit(gen, (250, 10))
        pygame.display.update()
        clock.tick(60)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
