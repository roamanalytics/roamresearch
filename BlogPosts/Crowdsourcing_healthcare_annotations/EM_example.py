"""
Simple implementation of Expectation-Maximization
for the task of judgments aggregation, with dummy data.
"""

CONVERGENCE_THRESHOLD = 0.01


class Question:
    
    def __init__(self):
        self.judgments = []
        self.prob = 0.5
        
    def estimate_probability(self):
        """
        Compute the max-likelihood estimate
        that the true response is 'True'.
        """
        true_term = 1.0  # p(j|r='T')
        false_term = 1.0 # p(j|r='F')
        
        for j in self.judgments:
            if j.response:
                true_term *= j.worker.tp_rate
                false_term *= j.worker.fp_rate
            else:
                false_term *= j.worker.tn_rate
                true_term *= j.worker.fn_rate 
                    
        self.prob = true_term / (true_term + false_term)
        return self.prob
    
    
class Worker:

    def __init__(self):
        self.judgments = []
        # Initial performance estimates
        self.tp_rate = 0.7
        self.fp_rate = 0.3
        self.tn_rate = 0.7
        self.fn_rate = 0.3
        self.precision = None
        
    def estimate_performance(self):
        tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
        for j in self.judgments:
            if j.response:
                tp += j.question.prob
                fp += 1.0 - j.question.prob
            else:
                fn += j.question.prob
                tn += 1.0 - j.question.prob

        # Small value added to denominators
        # to avoid dividing by zero 
        EPSILON = 1e-6
        self.tp_rate = tp / (tp + fn + EPSILON)
        self.fp_rate = fp / (fp + tn + EPSILON)
        self.tn_rate = tn / (tn + fp + EPSILON)
        self.fn_rate = fn / (fn + tp + EPSILON)
        self.precision = tp / (tp + fp + EPSILON)
    

class Judgment:
    
    def __init__(self, question, worker, response):
        self.question = question
        self.worker = worker
        self.response = response


def populate_data_structures(crowd_data):
    questions = [Question() for j in crowd_data]
    if crowd_data:
        workers = [Worker() for j in crowd_data[0]]
    else:
        workers = []

    for q_id, q_data in enumerate(crowd_data):
        for w_id, d in enumerate(q_data):
            if d is None:
                continue
            j = Judgment(questions[q_id], workers[w_id], d)
            j.question.judgments.append(j)
            j.worker.judgments.append(j)
    return questions, workers


def expectation_maximization(questions, workers):
    if not questions or not workers:
        return

    conv_delta = None
    it_count = 1
    while conv_delta is None or \
        conv_delta / len(questions) > CONVERGENCE_THRESHOLD:
        
        # Maximization
        conv_delta = 0.0
        for q in questions:
            conv_delta += abs(q.prob - q.estimate_probability())
        print('It. {}: {}'.format(it_count, conv_delta / len(questions)))
        it_count += 1
        
        # Expectation
        for w in workers:
            w.estimate_performance()


def print_results(questions, workers):
    print('')
    for i, q in enumerate(questions):
        print('question {} - {} ({})'
            .format(i+1, q.prob > 0.5, q.prob))
    
    print('')
    for i, w in enumerate(workers):
        if len(w.judgments) > 0:
            print('worker {} - pr: {}, re: {}'
                .format(i+1, w.precision, w.tp_rate))


if __name__ == "__main__":
    # Dummy data (5 questions, 4 workers)
    crowd_data = [
        [True, True, True, None],   # question 1
        [None, False, False, True], # question 2
        [True, True, False, None],  # question 3
        [False, None, False, True], # question 4
        [False, True, None, True]   # question 5
    ]
    questions, workers = populate_data_structures(crowd_data)
    expectation_maximization(questions, workers)
    print_results(questions, workers)
