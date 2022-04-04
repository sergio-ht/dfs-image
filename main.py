from __future__ import annotations
from collections import deque
from typing import Optional
from PIL import Image, ImageDraw
from dataclasses import dataclass, field
import random


RED = (237, 28, 36)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
START_COLOR = ORANGE = (255, 127, 39)
END_COLOR = BLUE = (63, 72, 204)


@dataclass
class Circle:
    x: int
    y: int
    radius: int


@dataclass(eq=True, frozen=True)
class Pixel:
    x: int
    y: int
    prev: Pixel = field(default=None, repr=False)

    def __eq__(self, other) -> bool:
        return isinstance(other, Pixel) and self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


def paint_circle(circle: Circle, color: tuple[int, int, int], img: Image) -> None:
    # get drawing object
    draw = ImageDraw.Draw(img)

    # define limits
    offset = 5
    top_left = (circle.x - circle.radius - offset, circle.y - circle.radius - offset)
    bottom_right = (
        circle.x + circle.radius + offset,
        circle.y + circle.radius + offset,
    )

    # draw
    draw.ellipse((top_left, bottom_right), fill=color, outline=None)


def paint_path(pixel: Pixel, color: tuple[int, int, int], img: Image) -> None:
    # get pixel map
    pixels_map = img.load()

    # paint path
    while pixel:
        pixels_map[pixel.x, pixel.y] = color
        pixel = pixel.prev


class Maze:
    def __init__(self, img: Image) -> None:
        self.img = img
        self.pixels = img.load()

    def find_start_end_points(self) -> None:
        for row in range(self.img.size[0]):
            for col in range(self.img.size[1]):
                if self.pixels[col, row] == START_COLOR:
                    # set start cirlce and pixel
                    self.start_circle = find_circle(col, row, self.pixels)
                    self.start_pixel = Pixel(self.start_circle.x, self.start_circle.y)

                    # erase circle from the image to avoid finding it again
                    paint_circle(circle=self.start_circle, color=WHITE, img=self.img)

                elif self.pixels[col, row] == END_COLOR:
                    # set end cirlce and pixel
                    self.end_circle = find_circle(col, row, self.pixels)
                    self.end_pixel = Pixel(self.end_circle.x, self.end_circle.y)

                    # erase circle from the image to avoid finding it again
                    paint_circle(circle=self.end_circle, color=WHITE, img=self.img)

    def bfs(self) -> Optional[Pixel]:
        # create visited set and queue
        visited = set()
        queue = deque([self.start_pixel])

        while queue:
            # get pixel
            pixel = queue.popleft()

            # check visited status
            if pixel in visited:
                continue
            visited.add(pixel)

            # check if it is the solution
            if pixel == self.end_pixel:
                self.end_pixel = pixel
                return self.end_pixel

            if pixel.x == self.end_pixel.x and pixel.y == self.end_pixel.y:
                print("SOLUTION FOUND")
                return (pixel.x, pixel.y)

            # add neighbors to queue
            for neighbor in get_neighbors(pixel, self.img):
                # check that it is not an obstacle
                x, y = neighbor
                if self.pixels[x, y] != WHITE:
                    continue
                new_pixel = Pixel(x=x, y=y, prev=pixel)
                queue.append(new_pixel)

        return None

    def dfs(self, max_its: Optional[int] = None) -> Optional[Pixel]:
        visited = set()
        stack = [self.start_pixel]

        while stack:

            # get pixel
            pixel = stack.pop()

            # check iterations
            if max_its:
                max_its -= 1
                if max_its <= 0:
                    return pixel

            # check visited status
            if pixel in visited:
                continue
            visited.add(pixel)

            # check if pixel == solution
            if pixel == self.end_pixel:
                self.end_pixel = pixel
                return self.end_pixel

            for neighbor in get_neighbors(pixel, self.img, shuffle=True):
                # check that it is not an obstacle
                x, y = neighbor
                if self.pixels[x, y] != WHITE:
                    continue
                new_pixel = Pixel(x=x, y=y, prev=pixel)
                if new_pixel in visited:
                    continue
                stack.append(new_pixel)

        return None


def find_circle(x: int, y: int, pixels) -> Circle:
    """Find and return the coordinates of the center and the radius of a circle
    Receive the coordinate of the leftmost pixel, of the top row of the circle
    """
    # vertical axis
    top = y
    while pixels[x, y + 1] != WHITE:
        y += 1
    y = (top + y) // 2

    # horizontal axis
    while pixels[x - 1, y] != WHITE:
        x -= 1
    left = x
    while pixels[x + 1, y] != WHITE:
        x += 1
    x = (left + x) // 2

    return Circle(x, y, y - top)


def get_neighbors(pixel: Pixel, img: Image, shuffle: bool = False) -> list[tuple[int, int]]:
    steps = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    neighbors = []

    for step in steps:
        x = pixel.x + step[0]
        y = pixel.y + step[1]

        if 0 <= x < img.size[0] and 0 <= y < img.size[1]:
            neighbors.append((x, y))

    if shuffle:
        random.shuffle(neighbors)

    return neighbors


def show_solution(pixel: Pixel, maze: Maze) -> None:
    """Paint the found path and the start and end circle"""

    # paint path
    paint_path(pixel, RED, maze.img)
    # paint circles
    paint_circle(maze.start_circle, START_COLOR, maze.img)
    paint_circle(maze.end_circle, END_COLOR, maze.img)
    # show img
    # maze.img.show()

    # save img
    maze.img.save(f"test-img-{random.randint(1, 1000)}.png")


def main():

    img_file = "img_6.png"
    maze = Maze(Image.open(img_file))
    maze.find_start_end_points()

    sol = maze.dfs()

    show_solution(sol, maze)


main()
