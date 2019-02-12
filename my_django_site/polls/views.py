from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.template import loader
from django.urls import reverse

from .models import Question, Choice

# Create your views here.
# def index(request):
#     latest_question_l = Question.objects.order_by('-pub_dt')[:5]
#     template = loader.get_template('polls/index.html')
#     context = {
#         'latest_question_l': latest_question_l,
#     }

#     return HttpResponse(template.render(context, request))

def index(request):
    latest_question_l = Question.objects.order_by('-pub_dt')[:5]
    context = {
        'latest_question_l':latest_question_l
    }
    return render(request, 'polls/index.html', context)


def detail(request, question_id):
    # try:
    #     question = Question.objects.get(pk=question_id)
    #     context = {'question':question}
    # except Question.DoesNotExist:
    #     raise Http404('question does not exist')
    question = get_object_or_404(Question, pk=question_id)
    context = {'question':question}

    return render(request, 'polls/detail.html', context)


def results(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    context = {'question': question}
    return render(request, 'polls/results.html', context)


def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)

    try:
        selection = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        context = {
            'question':question,
            'error_message':'please make selection'
        }
        return render(request, 'polls/detail.html', context)
    
    selection.votes += 1
    selection.save()

    return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
    