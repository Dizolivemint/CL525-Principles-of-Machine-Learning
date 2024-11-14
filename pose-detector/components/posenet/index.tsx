'use client'

import Script from 'next/script'
import { useEffect, useRef, useState } from 'react'

interface PoseKeypoint {
  score: number
  position: {
    x: number
    y: number
  }
}

interface Pose {
  pose: {
    keypoints: PoseKeypoint[]
  }
  skeleton: Array<[{ position: { x: number, y: number } }, { position: { x: number, y: number } }]>
}

export default function PoseNetClient() {
  const videoRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLDivElement>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isScriptsReady, setIsScriptsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [permissionStatus, setPermissionStatus] = useState<PermissionState | null>(null)

  // Check camera permission status
  useEffect(() => {
    const checkPermissions = async () => {
      try {
        const result = await navigator.permissions.query({ name: 'camera' as PermissionName })
        setPermissionStatus(result.state)
        
        result.addEventListener('change', () => {
          setPermissionStatus(result.state)
        })
      } catch (err) {
        console.warn('Permission API not supported, will try direct camera access')
      }
    }

    checkPermissions()
  }, [])

  useEffect(() => {
    if (!isScriptsReady || !window.p5 || !window.ml5) return

    let videoStream: any
    let poseNet: any
    let poses: Pose[] = []
    let sketch: any = null
    
    const initializeCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          } 
        })
        
        sketch = new window.p5((p: any) => {
          p.setup = () => {
            const canvas = p.createCanvas(640, 480)
            canvas.parent(canvasRef.current)
            
            videoStream = p.createVideo([])
            videoStream.elt.srcObject = stream
            videoStream.size(640, 480)
            videoStream.play()

            // Initialize PoseNet
            poseNet = window.ml5.poseNet(videoStream, {
              architecture: 'MobileNetV1',
              imageScaleFactor: 0.3,
              outputStride: 16,
              flipHorizontal: false,
              minConfidence: 0.5,
              maxPoseDetections: 5,
              scoreThreshold: 0.5,
              nmsRadius: 20,
              detectionType: 'single',
              multiplier: 0.75,
            }, () => {
              console.log('Model Loaded!')
              setIsLoading(false)
            })

            poseNet.on('pose', (results: Pose[]) => {
              poses = results
            })
          }

          p.draw = () => {
            if (videoStream) {
              p.image(videoStream, 0, 0)
              drawKeypoints()
              drawSkeleton()
            }
          }

          const drawKeypoints = () => {
            for (let i = 0; i < poses.length; i++) {
              const pose = poses[i].pose
              for (let j = 0; j < pose.keypoints.length; j++) {
                const keypoint = pose.keypoints[j]
                if (keypoint.score > 0.2) {
                  p.fill(255, 0, 0)
                  p.noStroke()
                  p.ellipse(keypoint.position.x, keypoint.position.y, 10, 10)
                }
              }
            }
          }

          const drawSkeleton = () => {
            for (let i = 0; i < poses.length; i++) {
              const skeleton = poses[i].skeleton
              for (let j = 0; j < skeleton.length; j++) {
                const partA = skeleton[j][0]
                const partB = skeleton[j][1]
                p.stroke(255, 0, 0)
                p.line(partA.position.x, partA.position.y, partB.position.x, partB.position.y)
              }
            }
          }
        })
      } catch (err: any) {
        if (err.name === 'NotAllowedError') {
          setError('Camera access denied. Please enable camera permissions.')
        } else if (err.name === 'NotFoundError') {
          setError('No camera found. Please connect a camera and try again.')
        } else {
          setError(`Error accessing camera: ${err.message}`)
        }
        setIsLoading(false)
      }
    }

    if (permissionStatus !== 'denied') {
      initializeCamera()
    } else {
      setError('Camera permission denied. Please enable camera access in your browser settings.')
      setIsLoading(false)
    }

    return () => {
      if (videoStream?.elt?.srcObject) {
        const tracks = videoStream.elt.srcObject.getTracks()
        tracks.forEach((track: MediaStreamTrack) => track.stop())
      }
      if (sketch) sketch.remove()
    }
  }, [isScriptsReady, permissionStatus])

  return (
    <div className="flex flex-col items-center gap-4">
      <Script
        src="https://cdn.jsdelivr.net/npm/p5@1.11.1/lib/p5.min.js"
        strategy="beforeInteractive"
      />
      <Script
        src="https://unpkg.com/ml5@1/dist/ml5.min.js"
        strategy="beforeInteractive"
        onLoad={() => setIsScriptsReady(true)}
      />
      
      <h1 className="text-2xl font-bold">PoseNet Demo</h1>
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
          <span className="block sm:inline">{error}</span>
        </div>
      )}
      {isLoading && !error ? (
        <div className="flex items-center gap-2">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-gray-900"></div>
          <p>Loading PoseNet...</p>
        </div>
      ) : (
        <div ref={canvasRef} className="border rounded-lg overflow-hidden shadow-lg"></div>
      )}
      {permissionStatus === 'prompt' && (
        <p className="text-sm text-gray-600">
          You'll be asked to allow camera access when starting the demo.
        </p>
      )}
    </div>
  )
}