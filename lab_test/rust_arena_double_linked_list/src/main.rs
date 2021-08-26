//rust如何实现双向链表？ - 姜哲的回答 - 知乎
//https://www.zhihu.com/question/54265676/answer/1679897001
//
use slotmap::{SlotMap, DefaultKey};

struct LinkedEntry<T> {
    value: T,
    prev: Option<DefaultKey>,
    next: Option<DefaultKey>,
}

pub struct LinkedList<T> {
    sm: SlotMap<DefaultKey, LinkedEntry<T>>,
    head: Option<DefaultKey>,
    tail: Option<DefaultKey>,
}

impl<T> LinkedList<T> {

    pub fn new() -> Self {
        Self{sm: SlotMap::new(), head: None, tail: None}
    }

    pub fn push_back(&mut self, elem: T) -> DefaultKey {
        let entry = LinkedEntry{value: elem, prev: self.tail, next: None};
        let key = self.sm.insert(entry);
        if self.head.is_none() {
            self.head = Some(key);
        }
        if let Some(tail) = self.tail {
            self.sm[tail].next = Some(key);
        }
        self.tail = Some(key);
        key
    }

    pub fn push_front(&mut self, elem: T) -> DefaultKey {
        let entry = LinkedEntry{value: elem, prev: None, next: self.head};
        let key = self.sm.insert(entry);
        if self.tail.is_none() {
            self.tail = Some(key);
        }
        if let Some(head) = self.head {
            self.sm[head].prev = Some(key);
        }
        self.head = Some(key);
        key
    }

    pub fn pop_back(&mut self) -> Option<T> {
        if let Some(tail) = self.tail {
            let entry = self.sm.remove(tail).unwrap();
            self.tail = entry.prev;

            if let Some(head) = self.head {
                if head == tail {
                    self.head = None;
                }
            }
            return Some(entry.value);
        }
        None
    }

    pub fn pop_front(&mut self) -> Option<T> {
        if let Some(head) = self.head {
            let entry = self.sm.remove(head).unwrap();
            self.head = entry.next;

            if let Some(tail) = self.tail {
                if tail == head {
                    self.tail = None;
                }
            }
            return Some(entry.value);
        }
        None
    }

    pub fn remove(&mut self, key: DefaultKey) -> Option<T> {
        if let Some(entry) = self.sm.remove(key) {
            if let Some(prev) = entry.prev {
                self.sm[prev].next = entry.next;
            }
            if let Some(next) = entry.next {
                self.sm[next].prev = entry.prev;
            }
            if let Some(head) = self.head {
                if head == key {
                    self.head = entry.next;
                }
            }
            if let Some(tail) = self.tail {
                if tail == key {
                    self.tail = entry.prev;
                }
            }
            return Some(entry.value)
        }
        None
    }

    pub fn get(&self, key: DefaultKey) -> Option<&T> {
        self.sm.get(key).map(|entry| &entry.value)
    }

    pub fn front(&self) -> Option<&T> {
        self.head.map(|head| &self.sm[head].value)
    }

    pub fn back(&self) -> Option<&T> {
        self.tail.map(|tail| &self.sm[tail].value)
    }
}

fn main()
{
    let mut list = LinkedList::new();
    list.push_front(0);
    list.push_front(1);
    list.push_front(2);
    list.push_front(3);
    print!("{:?} \r\n", list.front());
    print!("{:?} \r\n", list.pop_back());
    print!("{:?} \r\n", list.front());
}
