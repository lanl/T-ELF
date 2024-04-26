class BST(object):
    def __init__(self, x=None):
        self.value = x
        self.left = None
        self.right = None

    @classmethod
    def sorted_array_to_bst(node, nums):
        if not nums:
            return None
        
        mid_num = len(nums) // 2
        root = node(nums[mid_num])
        root.left = node.sorted_array_to_bst(nums[:mid_num])
        root.right = node.sorted_array_to_bst(nums[(mid_num + 1):])
        return root

    def preorder(self):
        yield self.value
        if self.left:
            yield from self.left.preorder()
        if self.right:
            yield from self.right.preorder()

    def postorder(self):
        yield self.value
        if self.right:
            yield from self.right.preorder()
        if self.left:
            yield from self.left.preorder()

