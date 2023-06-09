from models.base_op import MessageOp


class LastMessageOp(MessageOp):
    def __init__(self):
        super(LastMessageOp, self).__init__()
        self._aggr_type = "last"

    def combine(self, feat_list):
        return feat_list[-1]