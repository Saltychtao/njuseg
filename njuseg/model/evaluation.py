from functools import total_ordering

@total_ordering
class FScore(object):
    def __init__(self,correct=0,predcount=0,goldcount=0):
        self.correct = correct
        self.predcount = predcount
        self.goldcount = goldcount

    def precision(self):
        if self.predcount > 0:
            return (100.0 * self.correct) / self.predcount
        else:
            return 0.0

    def recall(self):
        if self.goldcount > 0:
            return (100.0 * self.correct) / self.goldcount
        else:
            return 0.0

    def fscore(self):
        precision = self.precision()
        recall = self.recall()
        if (precision + recall) > 0:
            return (2*precision*recall) / (precision+recall)
        else:
            return 0.0

    def __str__(self):
        precision = self.precision()
        recall = self.recall()
        fscore = self.fscore()
        return '(P = {:0.2f}, R = {:0.2f}, F = {:0.2f})'.format(
            precision,
            recall,
            fscore
        )

    def __iadd__(self, other):
        self.correct += other.correct
        self.predcount += other.predcount
        self.goldcount += other.goldcount
        return self

    def __add__(self, other):
        return FScore(self.correct+other.correct,
                      self.predcount+other.predcount,
                      self.goldcount+other.goldcount)

    def __eq__(self, other):
        return self.fscore() == other.fscore()

    def __lt__(self, other):
        return self.fscore() < other.fscore()

    @staticmethod
    def calc_span(sequence):
        span_set = set()
        begin = 0
        end = 0

        for i,t in enumerate(sequence):
            if t == 'S':
                span_set.add((i,i))
            elif t == 'B':
                begin = i
                end = i
            elif t == 'M':
                end += 1
            elif t == 'E':
                end += 1
                span_set.add((begin,end))
        return span_set

    @staticmethod
    def evaluate_BMSE(gold,pred):

        """
        Given two BMSE sequence, return the corresponding FScore
        :param gold:
        :param pred:
        :return:
        """

        assert len(gold) == len(pred)

        gold_spans = FScore.calc_span(gold)
        pred_spans = FScore.calc_span(pred)

        n_pred_spans = len(pred_spans)
        n_gold_spans = len(gold_spans)

        n_right_spans = 0
        for gs in gold_spans:
            if gs in pred_spans:
                n_right_spans += 1
        return FScore(correct=n_right_spans,predcount=n_pred_spans,goldcount=n_gold_spans)


@total_ordering
class Accuracy(object):
    def __init__(self, correct=0, count=0):
        self.correct = correct
        self.count = count


    def precision(self):
        if self.count > 0:
            return (100.0 * self.correct) / self.count
        else:
            return 0.0

    def __str__(self):
        precision = self.precision()
        return 'Precision = {:0.2f}'.format(
            precision,
        )

    def __iadd__(self, other):
        self.correct += other.correct
        self.count += other.count
        return self

    def __add__(self, other):
        return FScore(self.correct + other.correct,
                      self.count + other.count)

    def __eq__(self, other):
        return self.precision() == other.precision()

    def __lt__(self, other):
        return self.precision() < other.precision()

    @staticmethod
    def calc_precision(gold,pred):
        count = 0
        correct = 0
        for g,p in zip(gold,pred):
            if g == p:
                correct += 1
            count += 1
        return Accuracy(correct,count)
