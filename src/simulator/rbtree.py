class Node:
    def __init__(self, data, color='R'):
        self.data = data
        self.color = color  # 'R' for red, 'B' for black
        self.left = None
        self.right = None
        self.parent = None

    def __str__(self):
        return str(self.data)

    def __getattr__(self, item):
        # Delegate to the data attribute
        try:
            return getattr(self.data, item)
        except AttributeError:
            raise AttributeError(f"'Node' object has no attribute '{item}'")


class RedBlackTree:
    def __init__(self):
        self.TNULL = Node(data=None, color='B')
        self.root = self.TNULL
        self.size = 0

    def insert(self, key):
        new_node = Node(data=key)
        new_node.left = self.TNULL
        new_node.right = self.TNULL

        parent = None
        current = self.root
        self.size += 1

        while current != self.TNULL:
            parent = current
            if new_node.data < current.data:
                current = current.left
            else:
                current = current.right

        new_node.parent = parent
        if parent is None:
            self.root = new_node
        elif new_node.data < parent.data:
            parent.left = new_node
        else:
            parent.right = new_node

        if new_node.parent is None:
            new_node.color = 'B'
            return

        if new_node.parent.parent is None:
            return

        self.fix_insert(new_node)

    def fix_insert(self, k):
        while k.parent.color == 'R':
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left
                if u.color == 'R':
                    u.color = 'B'
                    k.parent.color = 'B'
                    k.parent.parent.color = 'R'
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self.right_rotate(k)
                    k.parent.color = 'B'
                    k.parent.parent.color = 'R'
                    self.left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right
                if u.color == 'R':
                    u.color = 'B'
                    k.parent.color = 'B'
                    k.parent.parent.color = 'R'
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        self.left_rotate(k)
                    k.parent.color = 'B'
                    k.parent.parent.color = 'R'
                    self.right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 'B'

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.TNULL:
            y.left.parent = x

        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.TNULL:
            y.right.parent = x

        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def transplant(self, u, v):
        if u.parent == None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def delete_node_helper(self, node, key):
        z = self.TNULL
        while node != self.TNULL:
            if node.data == key:
                z = node
            if node.data < key:
                node = node.right
            else:
                node = node.left

        if z == self.TNULL:
            print("Couldn't find key in the tree")
            return

        y = z
        y_original_color = y.color
        if z.left == self.TNULL:
            x = z.right
            self.transplant(z, z.right)
        elif z.right == self.TNULL:
            x = z.left
            self.transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self.transplant(y, y.right)
                y.right = z.right
                y.right.parent = y

            self.transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        if y_original_color == 'B':
            self.fix_delete(x)

    def fix_delete(self, x):
        while x != self.root and x.color == 'B':
            if x == x.parent.left:
                s = x.parent.right
                if s.color == 'R':
                    s.color = 'B'
                    x.parent.color = 'R'
                    self.left_rotate(x.parent)
                    s = x.parent.right

                if s.left.color == 'B' and s.right.color == 'B':
                    s.color = 'R'
                    x = x.parent
                else:
                    if s.right.color == 'B':
                        s.left.color = 'B'
                        s.color = 'R'
                        self.right_rotate(s)
                        s = x.parent.right

                    s.color = x.parent.color
                    x.parent.color = 'B'
                    s.right.color = 'B'
                    self.left_rotate(x.parent)
                    x = self.root
            else:
                s = x.parent.left
                if s.color == 'R':
                    s.color = 'B'
                    x.parent.color = 'R'
                    self.right_rotate(x.parent)
                    s = x.parent.left

                if s.left.color == 'B' and s.right.color == 'B':
                    s.color = 'R'
                    x = x.parent
                else:
                    if s.left.color == 'B':
                        s.right.color = 'B'
                        s.color = 'R'
                        self.left_rotate(s)
                        s = x.parent.left

                    s.color = x.parent.color
                    x.parent.color = 'B'
                    s.left.color = 'B'
                    self.right_rotate(x.parent)
                    x = self.root
        x.color = 'B'

    def minimum(self, node):
        while node != self.TNULL and node.left != self.TNULL:
            node = node.left
        return node

    def min(self) -> Node:
        return self.minimum(self.root)

    def maximum(self, node):
        while node != self.TNULL and node.right != self.TNULL:
            node = node.right
        return node

    def max(self):
        return self.maximum(self.root)

    def remove(self, key):
        self.size -= 1
        self.delete_node_helper(self.root, key)

    def pop(self, idx):
        self.size -= 1
        self.delete_node_helper(self.root, self.inorder()[idx].data)

    def inorder_helper(self, node):
        if node != self.TNULL:
            left = self.inorder_helper(node.left) or []
            right = self.inorder_helper(node.right) or []

            return left + [node.data] + right

    def inorder(self):
        return self.inorder_helper(self.root) or []

    def __len__(self):
        return self.size

    def search_helper(self, node: Node, key):
        if node == self.TNULL or key == node.data:
            return node
        if key < node.data:
            return self.search_helper(node.left, key)
        return self.search_helper(node.right, key)

    def search(self, key):
        return self.search_helper(self.root, key)

    def __getitem__(self, idx):
        return self.inorder()[idx]

    def __setitem__(self, key, value):
        self.inorder()[key] = value

    def __str__(self):
        return str(self.inorder())


if __name__ == "__main__":
    rbt = RedBlackTree()
    rbt.insert(20)
    rbt.insert(15)
    rbt.insert(25)
    rbt.insert(10)
    rbt.insert(5)
    rbt.insert(1)
    rbt.insert(30)

    print("Inorder before deletion:")
    print(rbt.inorder())

    rbt.remove(25)
    rbt.remove(20)
    rbt.remove(1)
    rbt.remove(15)

    print("\nInorder after deletion:")
    print(rbt.inorder())

    print("\nSize of the tree:", len(rbt))

    print("\nSearching for 10:", rbt.search(10).data)
    # first element O(log(h)) time complexity
    print("\nFirst element:", rbt[0])
