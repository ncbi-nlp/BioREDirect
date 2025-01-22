# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:50:26 2019

@author: laip2
"""



class AnnotationInfo:
    
    def __init__(self, position, length, text, ne_type):
        self.position = position
        self.length = length
        self.text = text
        self.ne_type = ne_type
        self.ids = set()
        self.corresponding_gene_id = ''
        self.orig_ne_type = '' # sometime ne_type is normalized eg "Variant" => 'Gene'
        self.corresponding_variant_ids = set()
        self.bioc_id = ''
        
class RelationInfo:
    
    def __init__(self):
        self.annotator  = ''
        self.updated_at = ''
        self.bioc_id = ''
        self.note = ''
        self.entities = []
        self.type = ''
        