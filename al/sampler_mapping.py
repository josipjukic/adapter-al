from al.albi import ActiveLearningBesovIndex
from al.bald.batch_bald import BatchBALDDropout
from al.discriminative import DiscriminativeRepresentationSampling
from al.kmeans import AntiKMeansSampler, KMeansSampler
from al.representation import (
    AntiMeanRepresentation,
    AntiRepresentation,
    MeanRepresentation,
    Representation,
)
from al.sampler import RandomSampler
from al.uncertainty import (
    AntiEntropySampler,
    AntiEntropySklearn,
    AntiMarginSampler,
    EntropyDropoutSampler,
    EntropySampler,
    LeastConfidentDropoutSampler,
    LeastConfidentSampler,
    MarginDropoutSampler,
    MarginSampler,
    MostConfidentSampler,
    EntropySklearn,
    MarginSklearn,
)
from al.core_set import CoreSet
from al.badge import BADGE, AntiBADGE


AL_SAMPLERS = {
    RandomSampler.name: RandomSampler,
    LeastConfidentSampler.name: LeastConfidentSampler,
    MarginSampler.name: MarginSampler,
    EntropySampler.name: EntropySampler,
    KMeansSampler.name: KMeansSampler,
    LeastConfidentDropoutSampler.name: LeastConfidentDropoutSampler,
    MarginDropoutSampler.name: MarginDropoutSampler,
    EntropyDropoutSampler.name: EntropyDropoutSampler,
    BADGE.name: BADGE,
    CoreSet.name: CoreSet,
    BatchBALDDropout.name: BatchBALDDropout,
    MostConfidentSampler.name: MostConfidentSampler,
    AntiMarginSampler.name: AntiMarginSampler,
    AntiEntropySampler.name: AntiEntropySampler,
    AntiKMeansSampler.name: AntiKMeansSampler,
    EntropySklearn.name: EntropySklearn,
    MarginSklearn.name: MarginSklearn,
    AntiEntropySklearn.name: AntiEntropySklearn,
    DiscriminativeRepresentationSampling.name: DiscriminativeRepresentationSampling,
    AntiBADGE.name: AntiBADGE,
    Representation.name: Representation,
    AntiRepresentation.name: AntiRepresentation,
    MeanRepresentation.name: MeanRepresentation,
    AntiMeanRepresentation.name: AntiMeanRepresentation,
    ActiveLearningBesovIndex.name: ActiveLearningBesovIndex,
    "besov": None,
}


def get_al_sampler(key):
    return AL_SAMPLERS[key]
