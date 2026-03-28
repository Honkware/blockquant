import { EventEmitter } from 'events';
import { db } from './db.js';
import { logger } from '../utils/logger.js';

class Queue extends EventEmitter {
  constructor() {
    super();
    this.running = false;
    this.current = null;
  }

  async add(job) {
    job.id = crypto.randomUUID();
    job.status = 'pending';
    job.created = Date.now();
    db.data.queue.push(job);
    await db.write();
    this.emit('added', job);
    if (!this.running) this.process();
    return job.id;
  }

  async process() {
    this.running = true;
    while (true) {
      const job = db.data.queue.find(j => j.status === 'pending');
      if (!job) break;
      this.current = job;
      job.status = 'running';
      job.started = Date.now();
      await db.write();
      this.emit('started', job);
      // processing happens here
      await new Promise(r => setTimeout(r, 1000));
      job.status = 'done';
      job.finished = Date.now();
      await db.write();
      this.emit('done', job);
    }
    this.running = false;
    this.current = null;
  }
}

export const queue = new Queue();
