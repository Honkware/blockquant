import { EventEmitter } from 'events';
import { spawn } from 'child_process';
import { db } from './db.js';
import { logger } from '../utils/logger.js';
import { path } from './workspace.js';

class Queue extends EventEmitter {
  constructor() {
    super();
    this.running = false;
    this.current = null;
  }

  async add(job) {
    job.id = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 5)}`;
    job.status = 'pending';
    job.created = Date.now();
    db.data.queue.push(job);
    await db.write();
    logger.info(`job ${job.id} queued`);
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
      
      logger.info(`processing job ${job.id}: ${job.model}`);
      
      try {
        const outputDir = path('output', job.id);
        const proc = spawn('python3', [
          'scripts/quantize.py',
          '--input', job.model,
          '--output', outputDir,
          '--bits', job.bits.toString()
        ], {
          env: { ...process.env, PYTHONPATH: 'exllamav3' }
        });

        await new Promise((resolve, reject) => {
          proc.on('close', (code) => {
            if (code === 0) resolve();
            else reject(new Error(`exit code ${code}`));
          });
          proc.on('error', reject);
        });

        job.status = 'done';
        job.output = outputDir;
        logger.info(`job ${job.id} complete`);
        
        // add exp
        const user = db.data.users[job.userId] || { exp: 0, jobs: 0 };
        user.exp += 10;
        user.jobs += 1;
        db.data.users[job.userId] = user;
        
      } catch (err) {
        job.status = 'failed';
        job.error = err.message;
        logger.error(`job ${job.id} failed: ${err.message}`);
      }
      
      job.finished = Date.now();
      await db.write();
      this.emit('done', job);
    }
    this.running = false;
    this.current = null;
  }
}

export const queue = new Queue();
