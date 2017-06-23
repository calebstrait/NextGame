# RedditAI.py
# by Caleb Strait
# Reddit filter that learns to recommend posts by user upvotes
# Runs on each redditai.stream page load via Flask app

from collections import namedtuple
import praw

def main(username, password, page):

    # Fetch credentials
    username = 'redditaistream'
    password = 'redditAI09373629!'

    # Connect to reddit.com API
    reddit = connect(username,password)

    # Get upvoted submissions's urls and titles
    upvoted = get_upvoted(username,reddit)

    # Create model from upvoted posts
    learned_model = learn(reddit,upvoted)

    # Fetch hot posts
    hot_posts = get_hot_posts(reddit)

    # Return recommendations
    return recommend(reddit,learned_model,hot_posts,upvoted,page)

def connect(username, password):
    # Fetch user from reddit.com API with API wrapper PRAW
    reddit = praw.Reddit(client_id='bXkbucDlO9O2PA',
        client_secret="4Ol3DKpME6bcVvWgxzvG7oLvWx0",
        user_agent='redditai.stream by /u/redditaistream',
        username=username, password=password)
    return reddit

def get_upvoted(username,reddit):

    # Get upvoted submissions's urls and titles
    this_redditor = praw.models.Redditor(reddit, name=str(username))
    return this_redditor.upvoted

def learn(reddit,upvoted):
    # [UNDER CONSTRUCTION]
    # Send potential posts through a neural network trained to find posts that you are likely to upvote:
    #
    #   Output Node:
    #     [probability of upvote]
    #
    #   Input Nodes:
    #     [top comment(s) sentiment]
    #     [score]
    #     [media type]
    #     if text:
    #       [post sentiment]
    #       [post length]
    #     if image:
    #       [image type/medium]
    #       [color palette]
    #
    # import tensorflow
    #
    # # Parameters
    # learning_rate = 0.001
    # training_epochs = 15
    # batch_size = 100
    # display_step = 1
    # num_hidden_1 = 256 # 1st hidden layer number of neurons
    # num_hidden_2 = 256 # 2nd hidden layer number of neurons
    # num_inputs = 784 # number of inputs
    # num_classes = 10 # total levels per input
    # x = tf.placeholder("float", [None, n_input])
    # y = tf.placeholder("float", [None, n_classes])
    #
    # # Store layers weight & bias
    # weights = {
    #     'h1': tensorflow.Variable(tensorflow.random_normal([n_input, n_hidden_1])),
    #     'h2': tensorflow.Variable(tensorflow.random_normal([n_hidden_1, n_hidden_2])),
    #     'out': tensorflow.Variable(tensorflow.random_normal([n_hidden_2, n_classes]))
    # }
    # biases = {
    #     'b1': tensorflow.Variable(tensorflow.random_normal([n_hidden_1])),
    #     'b2': tensorflow.Variable(tensorflow.random_normal([n_hidden_2])),
    #     'out': tensorflow.Variable(tensorflow.random_normal([n_classes]))
    # }
    #
    # def neural_net(x, weights, biases):
    #     # Hidden layer with RELU activation
    #     layer_1 = tensorflow.add(tensorflow.matmul(x, weights['h1']), biases['b1'])
    #     layer_1 = tensorflow.nn.relu(layer_1)
    #     # Hidden layer with RELU activation
    #     layer_2 = tensorflow.add(tensorflow.matmul(layer_1, weights['h2']), biases['b2'])
    #     layer_2 = tensorflow.nn.relu(layer_2)
    #     # Output layer with linear activation
    #     out = tensorflow.matmul(layer_2, weights['out']) + biases['out']
    #     return out_layer
    #
    # # Define model, cost function, & optimizer
    # prediction = neural_net(x, weights, biases)
    # cost = tensorflow.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # optimizer = tensorflow.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #
    # # Graph
    # with tensorflow.Session() as sess:
    #     sess.run(tensorflow.global_variables_initializer())
    #
    #     # Training cycle
    #     for epoch in range(training_epochs):
    #         avg_cost = 0.
    #         total_batch = int(mnist.train.num_examples/batch_size)
    #
    #         # Loop over all batches
    #         for i in range(total_batch):
    #             batch_x, batch_y = mnist.train.next_batch(batch_size)
    #
    #             # Run optimization op (backprop) and cost op (to get average loss value)
    #             _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
    #             avg_cost += c / total_batch
    #         # Display logs at each epoch
    #         if epoch % display_step == 0:
    #             print("Epoch:", '%04d' % (epoch+1), "cost=", \"{:.9f}".format(avg_cost))
    #
    #     # Test model
    #     correct_prediction = tensorflow.equal(tf.argmax(prediction, 1), tensorflow.argmax(y, 1))
    #     accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, "float"))
    #     print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    learned_model = 0

    return learned_model

def get_hot_posts(reddit):

    hot_posts = list()
    for submission in reddit.front.hot():
        hot_posts.append(submission)
    return hot_posts

def recommend(reddit,learned_model,hot_posts,upvoted,page):

    # Select posts to display
    num_results = 6
    to_recommend = hot_posts[(num_results*(page-1)):(num_results*page)]

    # Return urls and titles
    urls_dict = dict()
    titl_dict = dict()
    for r in to_recommend:
        num_results = num_results - 1
        urls_dict[2-num_results] = r.url
        submission = reddit.submission(id=r.id)
        titl_dict[2-num_results] = submission.title
    nt = namedtuple('nt', ['urls_dict', 'titl_dict'])
    return nt(urls_dict, titl_dict)

if __name__ == '__main__':
    main(username,password,page)
