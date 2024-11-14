'use client'

import Script from 'next/script'
import { useEffect, useRef, useState } from 'react'

export default function BodyPoseClient() {
  const canvasParentRef = useRef<HTMLDivElement>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isScriptsReady, setIsScriptsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  useEffect(() => {
    if (!isScriptsReady || !window.ml5 || !canvasParentRef.current) return

    const setupPose = async () => {
      try {
        console.log('Setting up BodyPose...')
        const bodyPose = await new Promise((resolve, reject) => {
          console.log('Initializing ML5 BodyPose...')
          const pose = window.ml5.bodyPose('MoveNet', {
            modelType: 'MULTIPOSE_LIGHTNING',
            enableSmoothing: true,
            minPoseScore: 0.25
          }, () => {
            console.log('BodyPose model loaded')
            resolve(pose)
          })
        })

        const connections = bodyPose.getSkeleton()
        console.log('Got skeleton connections:', connections)
        let poses: any[] = []
        let video: any

        new window.p5((p: any) => {
          p.setup = async () => {
            console.log('P5 setup starting...')
            const canvas = p.createCanvas(640, 480)
            canvas.parent(canvasParentRef.current)
            
            try {
              console.log('Requesting camera access...')
              const stream = await navigator.mediaDevices.getUserMedia({ 
                video: {
                  width: { ideal: 640 },
                  height: { ideal: 480 }
                } 
              })
              console.log('Camera access granted')
              video = p.createVideo([])
              video.elt.srcObject = stream
              video.size(640, 480)
              video.play()
              
              bodyPose.detectStart(video, (results: any[]) => {
                console.log('Poses detected:', results.length)
                poses = results
              })
            } catch (err) {
              console.error('Camera error:', err)
              setError('Camera access error: ' + (err as Error).message)
            }
            
            setIsLoading(false)
          }

          p.draw = () => {
            if (video) {
              p.image(video, 0, 0)
              
              poses.forEach((pose) => {
                // Draw keypoints with larger circles and labels
                pose.keypoints?.forEach((keypoint: any) => {
                  if (keypoint.confidence > 0.2) {
                    p.fill(255, 0, 0)
                    p.noStroke()
                    p.ellipse(keypoint.x, keypoint.y, 15, 15)
                    
                    p.fill(255)
                    p.stroke(0)
                    p.strokeWeight(2)
                    p.textSize(14)
                    p.text(keypoint.name || '', keypoint.x + 15, keypoint.y)
                  }
                })

                // Draw skeleton with thicker lines
                connections.forEach((connection) => {
                  const [p1, p2] = connection
                  const point1 = pose.keypoints?.[p1]
                  const point2 = pose.keypoints?.[p2]

                  if (point1?.confidence > 0.2 && point2?.confidence > 0.2) {
                    p.stroke(255, 0, 0)
                    p.strokeWeight(3)
                    p.line(point1.x, point1.y, point2.x, point2.y)
                  }
                })
              })
            }
          }
        })

      } catch (err) {
        console.error('Setup error:', err)
        setError((err as Error).message)
        setIsLoading(false)
      }
    }

    setupPose()
  }, [isScriptsReady])

  return (
    <div className="flex flex-col items-center gap-4">
      <Script
        src="https://cdn.jsdelivr.net/npm/p5@1.11.1/lib/p5.min.js"
        strategy="beforeInteractive"
      />
      <Script
        src="https://unpkg.com/ml5@1/dist/ml5.min.js"
        strategy="afterInteractive"
        onLoad={() => {
          console.log('ML5 loaded')
          setIsScriptsReady(true)
        }}
      />
      
      <h1 className="text-2xl font-bold">BodyPose Demo</h1>
      
      <div 
        ref={canvasParentRef}
        className="border rounded-lg overflow-hidden shadow-lg"
        style={{ width: '640px', height: '480px' }}
      />

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative">
          <span className="block sm:inline">{error}</span>
        </div>
      )}
      
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/80">
          <div className="flex items-center gap-2">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-gray-900"></div>
            <p>Loading BodyPose...</p>
          </div>
        </div>
      )}
    </div>
  )
}